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
class TFData2VecVisionPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Data2VecVisionConfig
    base_model_prefix = 'data2vec_vision'
    main_input_name = 'pixel_values'
    _keys_to_ignore_on_load_unexpected = ['relative_position_index']