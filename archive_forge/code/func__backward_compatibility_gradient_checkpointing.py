import copy
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mbart import MBartConfig
def _backward_compatibility_gradient_checkpointing(self):
    if self.supports_gradient_checkpointing and getattr(self.config, 'gradient_checkpointing', False):
        self.gradient_checkpointing_enable()