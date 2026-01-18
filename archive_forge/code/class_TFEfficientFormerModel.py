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
@add_start_docstrings('The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.', EFFICIENTFORMER_START_DOCSTRING)
class TFEfficientFormerModel(TFEfficientFormerPreTrainedModel):

    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.efficientformer = TFEfficientFormerMainLayer(config, name='efficientformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: Optional[tf.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        outputs = self.efficientformer(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'efficientformer', None) is not None:
            with tf.name_scope(self.efficientformer.name):
                self.efficientformer.build(None)