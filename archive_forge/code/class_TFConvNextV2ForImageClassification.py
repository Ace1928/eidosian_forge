from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_convnextv2 import ConvNextV2Config
@add_start_docstrings('\n    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', CONVNEXTV2_START_DOCSTRING)
class TFConvNextV2ForImageClassification(TFConvNextV2PreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.convnextv2 = TFConvNextV2MainLayer(config, name='convnextv2')
        self.classifier = keras.layers.Dense(units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), bias_initializer=keras.initializers.Zeros(), name='classifier')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: TFModelInputType | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        outputs = self.convnextv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convnextv2', None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_sizes[-1]])