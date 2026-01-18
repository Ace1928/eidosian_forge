import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
class GroundingDinoContrastiveEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_text_len = config.max_text_len

    def forward(self, vision_hidden_state: torch.FloatTensor, text_hidden_state: torch.FloatTensor, text_token_mask: torch.BoolTensor) -> torch.FloatTensor:
        output = vision_hidden_state @ text_hidden_state.transpose(-1, -2)
        output = output.masked_fill(~text_token_mask[:, None, :], float('-inf'))
        new_output = torch.full((*output.shape[:-1], self.max_text_len), float('-inf'), device=output.device)
        new_output[..., :output.shape[-1]] = output
        return new_output