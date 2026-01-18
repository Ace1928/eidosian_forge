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
@add_start_docstrings('\n    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', BERT_START_DOCSTRING)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['mlm___cls', 'nsp___cls', 'cls.predictions', 'cls.seq_relationship']
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = keras.layers.Dropout(rate=classifier_dropout)
        self.classifier = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_SEQ_CLASS_EXPECTED_OUTPUT, expected_loss=_SEQ_CLASS_EXPECTED_LOSS)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'bert', None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])