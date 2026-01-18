from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTTextModel(TFGroupViTPreTrainedModel):
    config_class = GroupViTTextConfig
    main_input_name = 'input_ids'

    def __init__(self, config: GroupViTTextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.groupvit = TFGroupViTTextMainLayer(config, name='groupvit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, TFGroupViTTextModel

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        outputs = self.groupvit(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'groupvit', None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)