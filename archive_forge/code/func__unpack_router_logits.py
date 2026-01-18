import copy
import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_switch_transformers import SwitchTransformersConfig
def _unpack_router_logits(self, router_outputs):
    total_router_logits = []
    total_expert_indexes = []
    for router_output in router_outputs:
        if len(router_output[0].shape) > 1:
            router_logits, expert_indexes = router_output
            total_router_logits.append(router_logits)
            total_expert_indexes.append(expert_indexes)
    return (torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1))