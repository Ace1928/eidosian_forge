import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
def _flatten_inputs(self, *inputs):
    """Flatten inputs' shape to (-1, input_shape[-1])"""
    flattened_inputs = []
    for tensor in inputs:
        if tensor is None:
            flattened_inputs.append(None)
        else:
            input_shape = tensor.shape
            if len(input_shape) > 2:
                tensor = tensor.view((-1, input_shape[-1]))
            flattened_inputs.append(tensor)
    return flattened_inputs