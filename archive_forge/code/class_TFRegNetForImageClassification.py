from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
@add_start_docstrings('\n    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', REGNET_START_DOCSTRING)
class TFRegNetForImageClassification(TFRegNetPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.regnet = TFRegNetMainLayer(config, name='regnet')
        self.classifier = [keras.layers.Flatten(), keras.layers.Dense(config.num_labels, name='classifier.1') if config.num_labels > 0 else tf.identity]

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: Optional[tf.Tensor]=None, labels: Optional[tf.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.regnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        flattened_output = self.classifier[0](pooled_output)
        logits = self.classifier[1](flattened_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'regnet', None) is not None:
            with tf.name_scope(self.regnet.name):
                self.regnet.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier[1].name):
                self.classifier[1].build([None, None, None, self.config.hidden_sizes[-1]])