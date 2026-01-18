import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wavlm import WavLMConfig
@staticmethod
def _compute_perplexity(probs):
    marginal_probs = probs.mean(dim=0)
    perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-07), dim=-1)).sum()
    return perplexity