import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
@add_start_docstrings('\n    The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', RWKV_START_DOCSTRING)
class RwkvForCausalLM(RwkvPreTrainedModel):
    _tied_weights_keys = ['head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def generate(self, *args, **kwargs):
        try:
            gen_output = super().generate(*args, **kwargs)
        except AttributeError as exc:
            if 'past_key_values' in str(exc):
                raise AttributeError("You tried to call `generate` with a decoding strategy that manipulates `past_key_values`. RWKV doesn't have that attribute, try another generation strategy instead. For the available generation strategies, check this doc: https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies")
            else:
                raise exc
        return gen_output

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and state is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs['state'] = state
        return model_inputs

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=RwkvCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, state: Optional[List[torch.FloatTensor]]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, RwkvCausalLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        rwkv_outputs = self.rwkv(input_ids, inputs_embeds=inputs_embeds, state=state, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = rwkv_outputs[0]
        logits = self.head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return (loss,) + output if loss is not None else output
        return RwkvCausalLMOutput(loss=loss, logits=logits, state=rwkv_outputs.state, hidden_states=rwkv_outputs.hidden_states, attentions=rwkv_outputs.attentions)