from __future__ import annotations
import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_tapas import TapasConfig
class TFTapasPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TapasConfig
    base_model_prefix = 'tapas'

    @property
    def input_signature(self):
        return {'input_ids': tf.TensorSpec((None, None), tf.int32, name='input_ids'), 'attention_mask': tf.TensorSpec((None, None), tf.float32, name='attention_mask'), 'token_type_ids': tf.TensorSpec((None, None, 7), tf.int32, name='token_type_ids')}