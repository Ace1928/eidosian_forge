import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_splinter import SplinterConfig
@add_start_docstrings('\n    Splinter Model for the recurring span selection task as done during the pretraining. The difference to the QA task\n    is that we do not have a question, but multiple question tokens that replace the occurrences of recurring spans\n    instead.\n    ', SPLINTER_START_DOCSTRING)
class SplinterForPreTraining(SplinterPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.splinter = SplinterModel(config)
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        self.question_token_id = config.question_token_id
        self.post_init()

    @add_start_docstrings_to_model_forward(SPLINTER_INPUTS_DOCSTRING.format('batch_size, num_questions, sequence_length'))
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.LongTensor]=None, end_positions: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, question_positions: Optional[torch.LongTensor]=None) -> Union[Tuple, SplinterForPreTrainingOutput]:
        """
        start_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        question_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            The positions of all question tokens. If given, start_logits and end_logits will be of shape `(batch_size,
            num_questions, sequence_length)`. If None, the first question token in each sequence in the batch will be
            the only one for which start_logits and end_logits are calculated and they will be of shape `(batch_size,
            sequence_length)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if question_positions is None and start_positions is not None and (end_positions is not None):
            raise TypeError('question_positions must be specified in order to calculate the loss')
        elif question_positions is None and input_ids is None:
            raise TypeError('question_positions must be specified when input_embeds is used')
        elif question_positions is None:
            question_positions = self._prepare_question_positions(input_ids)
        outputs = self.splinter(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        batch_size, sequence_length, dim = sequence_output.size()
        start_logits, end_logits = self.splinter_qass(sequence_output, question_positions)
        num_questions = question_positions.size(1)
        if attention_mask is not None:
            attention_mask_for_each_question = attention_mask.unsqueeze(1).expand(batch_size, num_questions, sequence_length)
            start_logits = start_logits + (1 - attention_mask_for_each_question) * torch.finfo(start_logits.dtype).min
            end_logits = end_logits + (1 - attention_mask_for_each_question) * torch.finfo(end_logits.dtype).min
        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_positions.clamp_(0, max(0, sequence_length - 1))
            end_positions.clamp_(0, max(0, sequence_length - 1))
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            start_loss = loss_fct(start_logits.view(batch_size * num_questions, sequence_length), start_positions.view(batch_size * num_questions))
            end_loss = loss_fct(end_logits.view(batch_size * num_questions, sequence_length), end_positions.view(batch_size * num_questions))
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        return SplinterForPreTrainingOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def _prepare_question_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, flat_positions = torch.where(input_ids == self.config.question_token_id)
        num_questions = torch.bincount(rows)
        positions = torch.full((input_ids.size(0), num_questions.max()), self.config.pad_token_id, dtype=torch.long, device=input_ids.device)
        cols = torch.cat([torch.arange(n) for n in num_questions])
        positions[rows, cols] = flat_positions
        return positions