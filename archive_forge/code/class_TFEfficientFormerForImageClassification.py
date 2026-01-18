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
@add_start_docstrings('\n    EfficientFormer Model transformer with an image classification head on top of pooled last hidden state, e.g. for\n    ImageNet.\n    ', EFFICIENTFORMER_START_DOCSTRING)
class TFEfficientFormerForImageClassification(TFEfficientFormerPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.efficientformer = TFEfficientFormerMainLayer(config, name='efficientformer')
        self.classifier = keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: Optional[tf.Tensor]=None, labels: Optional[tf.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tf.Tensor, TFImageClassifierOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.efficientformer(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.classifier(tf.reduce_mean(sequence_output, axis=-2))
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

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