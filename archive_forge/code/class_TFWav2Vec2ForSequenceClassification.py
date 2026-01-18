from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
class TFWav2Vec2ForSequenceClassification(TFWav2Vec2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = TFWav2Vec2MainLayer(config, name='wav2vec2')
        self.num_layers = config.num_hidden_layers + 1
        with tf.name_scope(self._name_scope()):
            if config.use_weighted_layer_sum:
                self.layer_weights = self.add_weight(shape=(self.num_layers,), initializer='ones', trainable=True, name='layer_weights')
        self.config = config
        self.projector = keras.layers.Dense(units=config.classifier_proj_size, name='projector')
        self.classifier = keras.layers.Dense(units=config.num_labels, activation=None, name='classifier')

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor.trainable = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for layer in self.wav2vec2.layers:
            layer.trainable = False

    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, labels: tf.Tensor | None=None, training: bool=False) -> TFSequenceClassifierOutput | Tuple[tf.Tensor]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = tf.stack(hidden_states, axis=1)
            norm_weights = tf.nn.softmax(self.layer_weights, axis=-1)
            hidden_states = tf.reduce_sum(hidden_states * tf.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(shape_list(hidden_states)[1], attention_mask)
            padding_mask_float = tf.cast(padding_mask, hidden_states.dtype)
            hidden_states = tf.multiply(hidden_states, tf.expand_dims(padding_mask_float, axis=-1))
            pooled_output = tf.divide(tf.reduce_sum(hidden_states, axis=1), tf.expand_dims(tf.reduce_sum(padding_mask_float, axis=1), axis=1))
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(tf.reshape(labels, [-1]), tf.reshape(logits, [-1, self.config.num_labels]))
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'wav2vec2', None) is not None:
            with tf.name_scope(self.wav2vec2.name):
                self.wav2vec2.build(None)
        if getattr(self, 'projector', None) is not None:
            with tf.name_scope(self.projector.name):
                self.projector.build([None, None, self.config.hidden_size])
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.classifier_proj_size])