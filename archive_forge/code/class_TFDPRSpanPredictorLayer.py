from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
class TFDPRSpanPredictorLayer(keras.layers.Layer):
    base_model_prefix = 'encoder'

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFDPREncoderLayer(config, name='encoder')
        self.qa_outputs = keras.layers.Dense(2, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')
        self.qa_classifier = keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='qa_classifier')

    @unpack_inputs
    def call(self, input_ids: tf.Tensor=None, attention_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False, training: bool=False) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        n_passages, sequence_length = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)[:2]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        relevance_logits = self.qa_classifier(sequence_output[:, 0, :])
        start_logits = tf.reshape(start_logits, [n_passages, sequence_length])
        end_logits = tf.reshape(end_logits, [n_passages, sequence_length])
        relevance_logits = tf.reshape(relevance_logits, [n_passages])
        if not return_dict:
            return (start_logits, end_logits, relevance_logits) + outputs[2:]
        return TFDPRReaderOutput(start_logits=start_logits, end_logits=end_logits, relevance_logits=relevance_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'qa_outputs', None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.encoder.embeddings_size])
        if getattr(self, 'qa_classifier', None) is not None:
            with tf.name_scope(self.qa_classifier.name):
                self.qa_classifier.build([None, None, self.encoder.embeddings_size])