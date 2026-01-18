from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
@add_start_docstrings('The bare Data2VecVision Model transformer outputting raw hidden-states without any specific head on top.', DATA2VEC_VISION_START_DOCSTRING)
class TFData2VecVisionModel(TFData2VecVisionPreTrainedModel):

    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool=False, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=add_pooling_layer, name='data2vec_vision')

    def get_input_embeddings(self):
        return self.data2vec_vision.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFData2VecVisionModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: TFModelInputType | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        """
        bool_masked_pos (`tf.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        outputs = self.data2vec_vision(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'data2vec_vision', None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)