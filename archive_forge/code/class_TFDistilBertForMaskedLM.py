from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_distilbert import DistilBertConfig
@add_start_docstrings('DistilBert Model with a `masked language modeling` head on top.', DISTILBERT_START_DOCSTRING)
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.distilbert = TFDistilBertMainLayer(config, name='distilbert')
        self.vocab_transform = keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='vocab_transform')
        self.act = get_tf_activation(config.activation)
        self.vocab_layer_norm = keras.layers.LayerNormalization(epsilon=1e-12, name='vocab_layer_norm')
        self.vocab_projector = TFDistilBertLMHead(config, self.distilbert.embeddings, name='vocab_projector')

    def get_lm_head(self):
        return self.vocab_projector

    def get_prefix_bias_name(self):
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.vocab_projector.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = distilbert_output[0]
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = self.act(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_logits)
        if not return_dict:
            output = (prediction_logits,) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'distilbert', None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        if getattr(self, 'vocab_transform', None) is not None:
            with tf.name_scope(self.vocab_transform.name):
                self.vocab_transform.build([None, None, self.config.dim])
        if getattr(self, 'vocab_layer_norm', None) is not None:
            with tf.name_scope(self.vocab_layer_norm.name):
                self.vocab_layer_norm.build([None, None, self.config.dim])
        if getattr(self, 'vocab_projector', None) is not None:
            with tf.name_scope(self.vocab_projector.name):
                self.vocab_projector.build(None)