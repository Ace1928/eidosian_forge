import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
    posemb_tok = posemb[:, :start_index]
    posemb_grid = posemb[0, start_index:]
    old_grid_size = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb