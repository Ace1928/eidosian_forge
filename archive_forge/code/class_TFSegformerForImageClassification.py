from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
@add_start_docstrings('\n    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden\n    states) e.g. for ImageNet.\n    ', SEGFORMER_START_DOCSTRING)
class TFSegformerForImageClassification(TFSegformerPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: SegformerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.segformer = TFSegformerMainLayer(config, name='segformer')
        self.classifier = keras.layers.Dense(config.num_labels, name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TFSequenceClassifierOutput]:
        outputs = self.segformer(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        batch_size = shape_list(sequence_output)[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])
        sequence_output = tf.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))
        sequence_output = tf.reduce_mean(sequence_output, axis=1)
        logits = self.classifier(sequence_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'segformer', None) is not None:
            with tf.name_scope(self.segformer.name):
                self.segformer.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])