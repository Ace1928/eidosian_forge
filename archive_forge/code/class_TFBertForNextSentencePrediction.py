from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_bert import BertConfig
@add_start_docstrings('Bert Model with a `next sentence prediction (classification)` head on top.', BERT_START_DOCSTRING)
class TFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
    _keys_to_ignore_on_load_unexpected = ['mlm___cls', 'cls.predictions']

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert')
        self.nsp = TFBertNSPHead(config, name='nsp___cls')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, next_sentence_label: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFNextSentencePredictorOutput, Tuple[tf.Tensor]]:
        """
        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = TFBertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        >>> assert logits[0][0] < logits[0][1]  # the next sentence was random
        ```"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        seq_relationship_scores = self.nsp(pooled_output=pooled_output)
        next_sentence_loss = None if next_sentence_label is None else self.hf_compute_loss(labels=next_sentence_label, logits=seq_relationship_scores)
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return (next_sentence_loss,) + output if next_sentence_loss is not None else output
        return TFNextSentencePredictorOutput(loss=next_sentence_loss, logits=seq_relationship_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'bert', None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, 'nsp', None) is not None:
            with tf.name_scope(self.nsp.name):
                self.nsp.build(None)