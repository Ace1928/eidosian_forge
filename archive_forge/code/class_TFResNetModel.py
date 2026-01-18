from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
@add_start_docstrings('The bare ResNet model outputting raw features without any specific head on top.', RESNET_START_DOCSTRING)
class TFResNetModel(TFResNetPreTrainedModel):

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.resnet = TFResNetMainLayer(config=config, name='resnet')

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        resnet_outputs = self.resnet(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return resnet_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'resnet', None) is not None:
            with tf.name_scope(self.resnet.name):
                self.resnet.build(None)