from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
@add_start_docstrings('\n    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', LONGFORMER_START_DOCSTRING)
class TFLongformerForSequenceClassification(TFLongformerPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name='longformer')
        self.classifier = TFLongformerClassificationHead(config, name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLongformerSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, global_attention_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFLongformerSequenceClassifierOutput, Tuple[tf.Tensor]]:
        if input_ids is not None and (not isinstance(input_ids, tf.Tensor)):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
        elif input_ids is not None:
            input_ids = tf.cast(input_ids, tf.int64)
        if attention_mask is not None and (not isinstance(attention_mask, tf.Tensor)):
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
        elif attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.int64)
        if global_attention_mask is not None and (not isinstance(global_attention_mask, tf.Tensor)):
            global_attention_mask = tf.convert_to_tensor(global_attention_mask, dtype=tf.int64)
        elif global_attention_mask is not None:
            global_attention_mask = tf.cast(global_attention_mask, tf.int64)
        if global_attention_mask is None and input_ids is not None:
            logger.warning_once('Initializing global attention on CLS token...')
            global_attention_mask = tf.zeros_like(input_ids)
            updates = tf.ones(shape_list(input_ids)[0], dtype=tf.int64)
            indices = tf.pad(tensor=tf.expand_dims(tf.range(shape_list(input_ids)[0], dtype=tf.int64), axis=1), paddings=[[0, 0], [0, 1]], constant_values=0)
            global_attention_mask = tf.tensor_scatter_nd_update(global_attention_mask, indices, updates)
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFLongformerSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'longformer', None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)