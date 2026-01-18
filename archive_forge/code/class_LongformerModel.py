import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longformer import LongformerConfig
@add_start_docstrings('The bare Longformer Model outputting raw hidden-states without any specific head on top.', LONGFORMER_START_DOCSTRING)
class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from [`RobertaModel`] and overwrote standard self-attention with longformer self-attention
    to provide the ability to process long sequences following the self-attention approach described in [Longformer:
    the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan.
    Longformer self-attention combines a local (sliding window) and global attention to extend to long documents
    without the O(n^2) increase in memory and compute.

    The self-attention module `LongformerSelfAttention` implemented here supports the combination of local and global
    attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated
    attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future
    release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA
    kernel to be memory and compute efficient.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, '`config.attention_window` has to be an even value'
            assert config.attention_window > 0, '`config.attention_window` has to be positive'
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}'
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        attention_window = self.config.attention_window if isinstance(self.config.attention_window, int) else max(self.config.attention_window)
        assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.warning_once(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.attention_window`: {attention_window}')
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=0)
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)
        return (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, global_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LongformerBaseModelOutputWithPooling]:
        """

        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import LongformerModel, AutoTokenizer

        >>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        >>> attention_mask = torch.ones(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        ... )  # initialize to local attention
        >>> global_attention_mask = torch.zeros(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        ... )  # initialize to global attention to be deactivated for all tokens
        >>> global_attention_mask[
        ...     :,
        ...     [
        ...         1,
        ...         4,
        ...         21,
        ...     ],
        ... ] = 1  # Set global attention to random tokens for the sake of this example
        >>> # Usually, set global attention based on the task. For example,
        >>> # classification: the <s> token
        >>> # QA: question tokens
        >>> # LM: potentially on the beginning of sentences and paragraphs
        >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        >>> sequence_output = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[:, 0, 0, :]
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, padding_len=padding_len, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return LongformerBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, global_attentions=encoder_outputs.global_attentions)