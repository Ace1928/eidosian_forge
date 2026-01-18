from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_gpt2 import GPT2Config
class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPT2Config
    base_model_prefix = 'transformer'
    _keys_to_ignore_on_load_unexpected = ['h.\\d+.attn.bias', 'h.\\d+.crossattention.bias']

    @property
    def input_signature(self):
        return {'input_ids': tf.TensorSpec((None, None), tf.int32, name='input_ids'), 'attention_mask': tf.TensorSpec((None, None), tf.int32, name='attention_mask')}