import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_hubert import HubertConfig
class HubertFeatureExtractor(HubertFeatureEncoder):

    def __init__(self, config):
        super().__init__(config)
        warnings.warn(f'The class `{self.__class__.__name__}` has been depreciated and will be removed in Transformers v5. Use `{self.__class__.__bases__[0].__name__}` instead.', FutureWarning)