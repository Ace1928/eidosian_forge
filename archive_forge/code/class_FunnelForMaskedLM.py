import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
@add_start_docstrings('Funnel Transformer Model with a `language modeling` head on top.', FUNNEL_START_DOCSTRING)
class FunnelForMaskedLM(FunnelPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.funnel = FunnelModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<mask>')
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.funnel(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = outputs[0]
        prediction_logits = self.lm_head(last_hidden_state)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)