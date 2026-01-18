import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
@add_start_docstrings('MobileBert Model with a `next sentence prediction (classification)` head on top.', MOBILEBERT_START_DOCSTRING)
class MobileBertForNextSentencePrediction(MobileBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.mobilebert = MobileBertModel(config)
        self.cls = MobileBertOnlyNSPHead(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, NextSentencePredictorOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`.

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MobileBertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = MobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        if 'next_sentence_label' in kwargs:
            warnings.warn('The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.', FutureWarning)
            labels = kwargs.pop('next_sentence_label')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mobilebert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        seq_relationship_score = self.cls(pooled_output)
        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), labels.view(-1))
        if not return_dict:
            output = (seq_relationship_score,) + outputs[2:]
            return (next_sentence_loss,) + output if next_sentence_loss is not None else output
        return NextSentencePredictorOutput(loss=next_sentence_loss, logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)