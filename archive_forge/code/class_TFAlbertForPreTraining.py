from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_albert import AlbertConfig
@add_start_docstrings('\n    Albert Model with two heads on top for pretraining: a `masked language modeling` head and a `sentence order\n    prediction` (classification) head.\n    ', ALBERT_START_DOCSTRING)
class TFAlbertForPreTraining(TFAlbertPreTrainedModel, TFAlbertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ['predictions.decoder.weight']

    def __init__(self, config: AlbertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.albert = TFAlbertMainLayer(config, name='albert')
        self.predictions = TFAlbertMLMHead(config, input_embeddings=self.albert.embeddings, name='predictions')
        self.sop_classifier = TFAlbertSOPHead(config, name='sop_classifier')

    def get_lm_head(self) -> keras.layers.Layer:
        return self.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFAlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, sentence_order_label: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFAlbertForPreTrainingOutput, Tuple[tf.Tensor]]:
        """
        Return:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFAlbertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
        >>> model = TFAlbertForPreTraining.from_pretrained("albert/albert-base-v2")

        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]
        >>> # Batch size 1
        >>> outputs = model(input_ids)

        >>> prediction_logits = outputs.prediction_logits
        >>> sop_logits = outputs.sop_logits
        ```"""
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(hidden_states=sequence_output)
        sop_scores = self.sop_classifier(pooled_output=pooled_output, training=training)
        total_loss = None
        if labels is not None and sentence_order_label is not None:
            d_labels = {'labels': labels}
            d_labels['sentence_order_label'] = sentence_order_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, sop_scores))
        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return TFAlbertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, sop_logits=sop_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'albert', None) is not None:
            with tf.name_scope(self.albert.name):
                self.albert.build(None)
        if getattr(self, 'predictions', None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        if getattr(self, 'sop_classifier', None) is not None:
            with tf.name_scope(self.sop_classifier.name):
                self.sop_classifier.build(None)