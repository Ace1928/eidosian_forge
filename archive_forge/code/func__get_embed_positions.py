import warnings
from typing import Optional, Tuple, Union
import torch
import torch.fx
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gptj import GPTJConfig
def _get_embed_positions(self, position_ids):
    embed_positions = self.embed_positions
    if embed_positions.device != position_ids.device:
        embed_positions = embed_positions.to(position_ids.device)
        self.embed_positions = embed_positions
    return embed_positions.repeat(position_ids.shape[0], 1, 1)