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
@add_start_docstrings('The bare ConvNextV2 model outputting raw features without any specific head on top.', CONVNEXTV2_START_DOCSTRING)
class TFConvNextV2Model(TFConvNextV2PreTrainedModel):

    def __init__(self, config: ConvNextV2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.convnextv2 = TFConvNextV2MainLayer(config, name='convnextv2')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: TFModelInputType | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        outputs = self.convnextv2(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return outputs[:]
        return TFBaseModelOutputWithPoolingAndNoAttention(last_hidden_state=outputs.last_hidden_state, pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convnextv2', None) is not None:
            with tf.name_scope(self.convnextv2.name):
                self.convnextv2.build(None)