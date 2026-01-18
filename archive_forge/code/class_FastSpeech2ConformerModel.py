import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
@add_start_docstrings('FastSpeech2Conformer Model.', FASTSPEECH2_CONFORMER_START_DOCSTRING)
class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    """
    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech'
    https://arxiv.org/abs/2006.04558. Instead of quantized pitch and energy, we use token-averaged value introduced in
    FastPitch: Parallel Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers instead of regular
    Transformers.
    """

    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_mel_bins = config.num_mel_bins
        self.hidden_size = config.hidden_size
        self.reduction_factor = config.reduction_factor
        self.stop_gradient_from_pitch_predictor = config.stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = config.stop_gradient_from_energy_predictor
        self.multilingual_model = config.num_languages is not None and config.num_languages > 1
        if self.multilingual_model:
            self.language_id_embedding = torch.nn.Embedding(config.num_languages, self.hidden_size)
        self.multispeaker_model = config.num_speakers is not None and config.num_speakers > 1
        if self.multispeaker_model:
            self.speaker_id_embedding = torch.nn.Embedding(config.num_speakers, config.hidden_size)
        self.speaker_embed_dim = config.speaker_embed_dim
        if self.speaker_embed_dim:
            self.projection = nn.Linear(config.hidden_size + self.speaker_embed_dim, config.hidden_size)
        self.encoder = FastSpeech2ConformerEncoder(config, config.encoder_config, use_encoder_input_layer=True)
        self.duration_predictor = FastSpeech2ConformerDurationPredictor(config)
        self.pitch_predictor = FastSpeech2ConformerVariancePredictor(config, num_layers=config.pitch_predictor_layers, num_chans=config.pitch_predictor_channels, kernel_size=config.pitch_predictor_kernel_size, dropout_rate=config.pitch_predictor_dropout)
        self.pitch_embed = FastSpeech2ConformerVarianceEmbedding(out_channels=self.hidden_size, kernel_size=config.pitch_embed_kernel_size, padding=(config.pitch_embed_kernel_size - 1) // 2, dropout_rate=config.pitch_embed_dropout)
        self.energy_predictor = FastSpeech2ConformerVariancePredictor(config, num_layers=config.energy_predictor_layers, num_chans=config.energy_predictor_channels, kernel_size=config.energy_predictor_kernel_size, dropout_rate=config.energy_predictor_dropout)
        self.energy_embed = FastSpeech2ConformerVarianceEmbedding(out_channels=self.hidden_size, kernel_size=config.energy_embed_kernel_size, padding=(config.energy_embed_kernel_size - 1) // 2, dropout_rate=config.energy_embed_dropout)
        self.decoder = FastSpeech2ConformerEncoder(config, config.decoder_config, use_encoder_input_layer=False)
        self.speech_decoder_postnet = FastSpeech2ConformerSpeechDecoderPostnet(config)
        self.criterion = FastSpeech2ConformerLoss(config)
        self.post_init()

    @replace_return_docstrings(output_type=FastSpeech2ConformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor]=None, spectrogram_labels: Optional[torch.FloatTensor]=None, duration_labels: Optional[torch.LongTensor]=None, pitch_labels: Optional[torch.FloatTensor]=None, energy_labels: Optional[torch.FloatTensor]=None, speaker_ids: Optional[torch.LongTensor]=None, lang_ids: Optional[torch.LongTensor]=None, speaker_embedding: Optional[torch.FloatTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None) -> Union[Tuple, FastSpeech2ConformerModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*, defaults to `None`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*, defaults to `None`):
                Batch of padded durations.
            pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged pitch.
            energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged energy.
            speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Speaker ids used to condition features of speech output by the model.
            lang_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Language ids used to condition features of speech output by the model.
            speaker_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*, defaults to `None`):
                Embedding containing conditioning signals for the features of the speech.
            return_dict (`bool`, *optional*, defaults to `None`):
                Whether or not to return a [`FastSpeech2ConformerModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*, defaults to `None`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `None`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     FastSpeech2ConformerTokenizer,
        ...     FastSpeech2ConformerModel,
        ...     FastSpeech2ConformerHifiGan,
        ... )

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        >>> output_dict = model(input_ids, return_dict=True)
        >>> spectrogram = output_dict["spectrogram"]

        >>> vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
        >>> waveform = vocoder(spectrogram)
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        has_missing_labels = spectrogram_labels is None or duration_labels is None or pitch_labels is None or (energy_labels is None)
        if self.training and has_missing_labels:
            raise ValueError('All labels must be provided to run in training mode.')
        text_masks = attention_mask.unsqueeze(-2)
        encoder_outputs = self.encoder(input_ids, text_masks, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if self.multispeaker_model and speaker_ids is not None:
            speaker_id_embeddings = self.speaker_id_embedding(speaker_ids.view(-1))
            hidden_states = hidden_states + speaker_id_embeddings.unsqueeze(1)
        if self.multilingual_model and lang_ids is not None:
            language_id_embbedings = self.language_id_embedding(lang_ids.view(-1))
            hidden_states = hidden_states + language_id_embbedings.unsqueeze(1)
        if self.speaker_embed_dim is not None and speaker_embedding is not None:
            embeddings_expanded = nn.functional.normalize(speaker_embedding).unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            hidden_states = self.projection(torch.cat([hidden_states, embeddings_expanded], dim=-1))
        duration_mask = ~attention_mask.bool()
        if self.stop_gradient_from_pitch_predictor:
            pitch_predictions = self.pitch_predictor(hidden_states.detach(), duration_mask.unsqueeze(-1))
        else:
            pitch_predictions = self.pitch_predictor(hidden_states, duration_mask.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            energy_predictions = self.energy_predictor(hidden_states.detach(), duration_mask.unsqueeze(-1))
        else:
            energy_predictions = self.energy_predictor(hidden_states, duration_mask.unsqueeze(-1))
        duration_predictions = self.duration_predictor(hidden_states)
        duration_predictions = duration_predictions.masked_fill(duration_mask, 0.0)
        if not self.training:
            embedded_pitch_curve = self.pitch_embed(pitch_predictions)
            embedded_energy_curve = self.energy_embed(energy_predictions)
            hidden_states = hidden_states + embedded_energy_curve + embedded_pitch_curve
            hidden_states = length_regulator(hidden_states, duration_predictions, self.config.speaking_speed)
        else:
            embedded_pitch_curve = self.pitch_embed(pitch_labels)
            embedded_energy_curve = self.energy_embed(energy_labels)
            hidden_states = hidden_states + embedded_energy_curve + embedded_pitch_curve
            hidden_states = length_regulator(hidden_states, duration_labels)
        if not self.training:
            hidden_mask = None
        else:
            spectrogram_mask = (spectrogram_labels != -100).any(dim=-1)
            spectrogram_mask = spectrogram_mask.int()
            if self.reduction_factor > 1:
                length_dim = spectrogram_mask.shape[1] - spectrogram_mask.shape[1] % self.reduction_factor
                spectrogram_mask = spectrogram_mask[:, :, :length_dim]
            hidden_mask = spectrogram_mask.unsqueeze(-2)
        decoder_outputs = self.decoder(hidden_states, hidden_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        outputs_before_postnet, outputs_after_postnet = self.speech_decoder_postnet(decoder_outputs[0])
        loss = None
        if self.training:
            loss_duration_mask = ~duration_mask
            loss_spectrogram_mask = spectrogram_mask.unsqueeze(-1).bool()
            loss = self.criterion(outputs_after_postnet=outputs_after_postnet, outputs_before_postnet=outputs_before_postnet, duration_outputs=duration_predictions, pitch_outputs=pitch_predictions, energy_outputs=energy_predictions, spectrogram_labels=spectrogram_labels, duration_labels=duration_labels, pitch_labels=pitch_labels, energy_labels=energy_labels, duration_mask=loss_duration_mask, spectrogram_mask=loss_spectrogram_mask)
        if not return_dict:
            postnet_outputs = (outputs_after_postnet,)
            audio_feature_predictions = (duration_predictions, pitch_predictions, energy_predictions)
            outputs = postnet_outputs + encoder_outputs + decoder_outputs[1:] + audio_feature_predictions
            return (loss,) + outputs if loss is not None else outputs
        return FastSpeech2ConformerModelOutput(loss=loss, spectrogram=outputs_after_postnet, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, duration_outputs=duration_predictions, pitch_outputs=pitch_predictions, energy_outputs=energy_predictions)