import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
@add_start_docstrings('\n    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden\n    state and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.\n\n    .. warning::\n            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet\n            supported.\n    ', EFFICIENTFORMER_START_DOCSTRING)
class TFEfficientFormerForImageClassificationWithTeacher(TFEfficientFormerPreTrainedModel):

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.efficientformer = TFEfficientFormerMainLayer(config, name='efficientformer')
        self.classifier = keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='classifier')
        self.distillation_classifier = keras.layers.Dense(config.num_labels, name='distillation_classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='distillation_classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFEfficientFormerForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: Optional[tf.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFEfficientFormerForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if training:
            raise Exception('This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported.')
        outputs = self.efficientformer(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        cls_logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))
        distillation_logits = self.distillation_classifier(tf.reduce_mean(sequence_output, axis=-2))
        logits = (cls_logits + distillation_logits) / 2
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output
        return TFEfficientFormerForImageClassificationWithTeacherOutput(logits=logits, cls_logits=cls_logits, distillation_logits=distillation_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'efficientformer', None) is not None:
            with tf.name_scope(self.efficientformer.name):
                self.efficientformer.build(None)
        if getattr(self, 'classifier', None) is not None:
            if hasattr(self.classifier, 'name'):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.hidden_sizes[-1]])
        if getattr(self, 'distillation_classifier', None) is not None:
            if hasattr(self.distillation_classifier, 'name'):
                with tf.name_scope(self.distillation_classifier.name):
                    self.distillation_classifier.build([None, None, self.config.hidden_sizes[-1]])