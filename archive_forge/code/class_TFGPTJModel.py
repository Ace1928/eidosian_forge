from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig
@add_start_docstrings('The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.', GPTJ_START_DOCSTRING)
class TFGPTJModel(TFGPTJPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPTJMainLayer(config, name='transformer')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        """
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """
        outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)