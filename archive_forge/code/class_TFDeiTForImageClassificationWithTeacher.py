from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_deit import DeiTConfig
@add_start_docstrings('\n    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of\n    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.\n\n    .. warning::\n\n            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet\n            supported.\n    ', DEIT_START_DOCSTRING)
class TFDeiTForImageClassificationWithTeacher(TFDeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=False, name='deit')
        self.cls_classifier = keras.layers.Dense(config.num_labels, name='cls_classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='cls_classifier')
        self.distillation_classifier = keras.layers.Dense(config.num_labels, name='distillation_classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='distillation_classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFDeiTForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFDeiTForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
        logits = (cls_logits + distillation_logits) / 2
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output
        return TFDeiTForImageClassificationWithTeacherOutput(logits=logits, cls_logits=cls_logits, distillation_logits=distillation_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'deit', None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)
        if getattr(self, 'cls_classifier', None) is not None:
            with tf.name_scope(self.cls_classifier.name):
                self.cls_classifier.build([None, None, self.config.hidden_size])
        if getattr(self, 'distillation_classifier', None) is not None:
            with tf.name_scope(self.distillation_classifier.name):
                self.distillation_classifier.build([None, None, self.config.hidden_size])