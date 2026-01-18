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
@add_start_docstrings('\n    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a\n    `next sentence prediction (classification)` head.\n    ', MOBILEBERT_START_DOCSTRING)
class MobileBertForPreTraining(MobileBertPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        self.mobilebert = MobileBertModel(config)
        self.cls = MobileBertPreTrainingHeads(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddigs):
        self.cls.predictions.decoder = new_embeddigs

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None) -> nn.Embedding:
        self.cls.predictions.dense = self._get_resized_lm_head(self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True)
        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=MobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, next_sentence_label: Optional[torch.LongTensor]=None, output_attentions: Optional[torch.FloatTensor]=None, output_hidden_states: Optional[torch.FloatTensor]=None, return_dict: Optional[torch.FloatTensor]=None) -> Union[Tuple, MobileBertForPreTrainingOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MobileBertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = MobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mobilebert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return MobileBertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)