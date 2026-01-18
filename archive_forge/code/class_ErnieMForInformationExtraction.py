import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig
@add_start_docstrings('ErnieMForInformationExtraction is a Ernie-M Model with two linear layer on top of the hidden-states output to\n    compute `start_prob` and `end_prob`, designed for Universal Information Extraction.', ERNIE_M_START_DOCSTRING)
class ErnieMForInformationExtraction(ErnieMPreTrainedModel):

    def __init__(self, config):
        super(ErnieMForInformationExtraction, self).__init__(config)
        self.ernie_m = ErnieMModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.post_init()

    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=True) -> Union[Tuple[torch.FloatTensor], QuestionAnsweringModelOutput]:
        """
        start_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for position (index) for computing the start_positions loss. Position outside of the sequence are
            not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) for computing the end_positions loss. Position outside of the sequence are not
            taken into account for computing the loss.
        """
        result = self.ernie_m(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if return_dict:
            sequence_output = result.last_hidden_state
        elif not return_dict:
            sequence_output = result[0]
        start_logits = self.linear_start(sequence_output)
        start_logits = start_logits.squeeze(-1)
        end_logits = self.linear_end(sequence_output)
        end_logits = end_logits.squeeze(-1)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = BCEWithLogitsLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            return tuple((i for i in [total_loss, start_logits, end_logits, result.hidden_states, result.attentions] if i is not None))
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=result.hidden_states, attentions=result.attentions)