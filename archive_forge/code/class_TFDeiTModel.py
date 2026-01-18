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
@add_start_docstrings('The bare DeiT Model transformer outputting raw hidden-states without any specific head on top.', DEIT_START_DOCSTRING)
class TFDeiTModel(TFDeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.deit = TFDeiTMainLayer(config, add_pooling_layer=add_pooling_layer, use_mask_token=use_mask_token, name='deit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        outputs = self.deit(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'deit', None) is not None:
            with tf.name_scope(self.deit.name):
                self.deit.build(None)