import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTaskModel(nn.Module):

    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.task_mlp = OneFormerMLPPredictionHead(config.task_seq_len, config.hidden_dim, config.hidden_dim, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        task_tokens = self.task_mlp(inputs)
        return task_tokens