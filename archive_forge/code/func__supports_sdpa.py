from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_vipllava import VipLlavaConfig
@property
def _supports_sdpa(self):
    """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
    return self.language_model._supports_sdpa