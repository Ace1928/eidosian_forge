from __future__ import annotations
import math
import random
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig
class TFXGLMPreTrainedModel(TFPreTrainedModel):
    config_class = XGLMConfig
    base_model_prefix = 'model'