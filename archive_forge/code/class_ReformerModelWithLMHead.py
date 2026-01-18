import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
@add_start_docstrings('Reformer Model with a `language modeling` head on top.', REFORMER_START_DOCSTRING)
class ReformerModelWithLMHead(ReformerPreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder.weight', 'lm_head.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        assert config.is_decoder, 'If you want to use `ReformerModelWithLMHead` make sure that `is_decoder=True`.'
        assert 'local' not in self.config.attn_layers or config.local_num_chunks_after == 0, f'If causal mask is enabled, make sure that `config.local_num_chunks_after` is set to 0 and not {config.local_num_chunks_after}.'
        assert 'lsh' not in self.config.attn_layers or config.lsh_num_chunks_after == 0, f'If causal mask is enabled, make sure that `config.lsh_num_chunks_after` is set to 1 and not {config.lsh_num_chunks_after}.'
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, num_hashes: Optional[int]=None, past_buckets_states: Optional[List[Tuple[torch.Tensor]]]=None, use_cache: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, CausalLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
                config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
                labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        reformer_outputs = self.reformer(input_ids, position_ids=position_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, num_hashes=num_hashes, past_buckets_states=past_buckets_states, use_cache=use_cache, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return ReformerModelWithLMHeadOutput(loss=loss, logits=logits, past_buckets_states=reformer_outputs.past_buckets_states, hidden_states=reformer_outputs.hidden_states, attentions=reformer_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, num_hashes=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        inputs_dict = {'input_ids': input_ids, 'past_buckets_states': past_key_values, 'use_cache': use_cache, 'num_hashes': num_hashes}
        return inputs_dict

    def _reorder_cache(self, past_key_values, beam_idx):
        reord_past_buckets_states = []
        for layer_past in past_key_values:
            if layer_past[0] is not None:
                reord_buckets = layer_past[0].index_select(0, beam_idx.to(layer_past[0].device))
            else:
                reord_buckets = None
            reord_hidden_states = layer_past[1].index_select(0, beam_idx.to(layer_past[1].device))
            reord_past_buckets_states.append((reord_buckets, reord_hidden_states))
        return reord_past_buckets_states