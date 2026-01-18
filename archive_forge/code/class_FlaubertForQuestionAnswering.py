import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary, SQuADHead
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flaubert import FlaubertConfig
class FlaubertForQuestionAnswering(FlaubertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.qa_outputs = SQuADHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=FlaubertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, langs: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, lengths: Optional[torch.Tensor]=None, cache: Optional[Dict[str, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, start_positions: Optional[torch.Tensor]=None, end_positions: Optional[torch.Tensor]=None, is_impossible: Optional[torch.Tensor]=None, cls_index: Optional[torch.Tensor]=None, p_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, FlaubertForQuestionAnsweringOutput]:
        """
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the classification token to use as input for computing plausibility of the
            answer.
        p_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
            masked. 0.0 mean token is not masked.

        Returns:

        Example:

        ```python
        >>> from transformers import XLMTokenizer, XLMForQuestionAnswering
        >>> import torch

        >>> tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-mlm-en-2048")
        >>> model = XLMForQuestionAnswering.from_pretrained("FacebookAI/xlm-mlm-en-2048")

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
        ...     0
        ... )  # Batch size 1
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        output = transformer_outputs[0]
        outputs = self.qa_outputs(output, start_positions=start_positions, end_positions=end_positions, cls_index=cls_index, is_impossible=is_impossible, p_mask=p_mask, return_dict=return_dict)
        if not return_dict:
            return outputs + transformer_outputs[1:]
        return FlaubertForQuestionAnsweringOutput(loss=outputs.loss, start_top_log_probs=outputs.start_top_log_probs, start_top_index=outputs.start_top_index, end_top_log_probs=outputs.end_top_log_probs, end_top_index=outputs.end_top_index, cls_logits=outputs.cls_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)