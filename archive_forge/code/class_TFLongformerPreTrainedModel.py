from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
class TFLongformerPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongformerConfig
    base_model_prefix = 'longformer'

    @property
    def input_signature(self):
        sig = super().input_signature
        sig['global_attention_mask'] = tf.TensorSpec((None, None), tf.int32, name='global_attention_mask')
        return sig