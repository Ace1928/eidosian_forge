import collections
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def _cal_1d_pos_emb(self, position_ids):
    rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
    rel_pos = self.relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
    rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
    rel_pos = rel_pos.contiguous()
    return rel_pos