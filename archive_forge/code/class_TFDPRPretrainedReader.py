from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
class TFDPRPretrainedReader(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    base_model_prefix = 'reader'