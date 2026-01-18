from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_informer import InformerConfig
@add_start_docstrings('The bare Informer Model outputting raw hidden-states without any specific head on top.', INFORMER_START_DOCSTRING)
class InformerModel(InformerPreTrainedModel):

    def __init__(self, config: InformerConfig):
        super().__init__(config)
        if config.scaling == 'mean' or config.scaling is True:
            self.scaler = InformerMeanScaler(config)
        elif config.scaling == 'std':
            self.scaler = InformerStdScaler(config)
        else:
            self.scaler = InformerNOPScaler(config)
        if config.num_static_categorical_features > 0:
            self.embedder = InformerFeatureEmbedder(cardinalities=config.cardinality, embedding_dims=config.embedding_dimension)
        self.encoder = InformerEncoder(config)
        self.decoder = InformerDecoder(config)
        self.post_init()

    @property
    def _past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(self, sequence: torch.Tensor, subsequences_length: int, shift: int=0) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence. Returns a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices), containing lagged subsequences. Specifically, lagged[i,
            j, :, k] = sequence[i, -indices[k]-S+j, :].

        Args:
            sequence: Tensor
                The sequence from which lagged subsequences should be extracted. Shape: (N, T, C).
            subsequences_length : int
                Length of the subsequences to be extracted.
            shift: int
                Shift the lags by this amount back.
        """
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.config.lags_sequence]
        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(f'lags cannot go further than history length, found lag {max(indices)} while history length is only {sequence_length}')
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def create_network_inputs(self, past_values: torch.Tensor, past_time_features: torch.Tensor, static_categorical_features: Optional[torch.Tensor]=None, static_real_features: Optional[torch.Tensor]=None, past_observed_mask: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, future_time_features: Optional[torch.Tensor]=None):
        time_feat = torch.cat((past_time_features[:, self._past_length - self.config.context_length:, ...], future_time_features), dim=1) if future_values is not None else past_time_features[:, self._past_length - self.config.context_length:, ...]
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        context = past_values[:, -self.config.context_length:]
        observed_context = past_observed_mask[:, -self.config.context_length:]
        _, loc, scale = self.scaler(context, observed_context)
        inputs = (torch.cat((past_values, future_values), dim=1) - loc) / scale if future_values is not None else (past_values - loc) / scale
        log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
        log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)
        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        if static_categorical_features is not None:
            embedded_cat = self.embedder(static_categorical_features)
            static_feat = torch.cat((embedded_cat, static_feat), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)
        subsequences_length = self.config.context_length + self.config.prediction_length if future_values is not None else self.config.context_length
        lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(f'input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match')
        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)
        return (transformer_inputs, loc, scale, static_feat)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(INFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, past_values: torch.Tensor, past_time_features: torch.Tensor, past_observed_mask: torch.Tensor, static_categorical_features: Optional[torch.Tensor]=None, static_real_features: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, future_time_features: Optional[torch.Tensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.Tensor]=None, decoder_head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[List[torch.FloatTensor]]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, use_cache: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Seq2SeqTSModelOutput, Tuple]:
        """
        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import InformerModel

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = InformerModel.from_pretrained("huggingface/informer-tourism-monthly")

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_inputs, loc, scale, static_feat = self.create_network_inputs(past_values=past_values, past_time_features=past_time_features, past_observed_mask=past_observed_mask, static_categorical_features=static_categorical_features, static_real_features=static_real_features, future_values=future_values, future_time_features=future_time_features)
        if encoder_outputs is None:
            enc_input = transformer_inputs[:, :self.config.context_length, ...]
            encoder_outputs = self.encoder(inputs_embeds=enc_input, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        dec_input = transformer_inputs[:, self.config.context_length:, ...]
        decoder_outputs = self.decoder(inputs_embeds=dec_input, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs + (loc, scale, static_feat)
        return Seq2SeqTSModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, loc=loc, scale=scale, static_features=static_feat)