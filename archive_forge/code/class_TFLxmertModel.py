from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
@add_start_docstrings('The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.', LXMERT_START_DOCSTRING)
class TFLxmertModel(TFLxmertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.lxmert = TFLxmertMainLayer(config, name='lxmert')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLxmertModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, visual_feats: tf.Tensor | None=None, visual_pos: tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, visual_attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFLxmertModelOutput]:
        outputs = self.lxmert(input_ids, visual_feats, visual_pos, attention_mask, visual_attention_mask, token_type_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict, training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'lxmert', None) is not None:
            with tf.name_scope(self.lxmert.name):
                self.lxmert.build(None)