from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..bert.modeling_bert import BertModel
from .configuration_dpr import DPRConfig
class DPRPretrainedReader(DPRPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = 'span_predictor'