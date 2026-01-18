from itertools import chain
from operator import getitem
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union
from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param
from .match_utils import apply_match, MatchAllNode
from .prune_functions import (
def _get_default_structured_pruning_patterns() -> Dict[Tuple[Union[Type[nn.Module], Callable, MatchAllNode, str], ...], Callable[..., None]]:
    """
    Returns the patterns for conv2d / linear conversion for each element in the activation functions/modules defined above.
    """
    patterns: Dict[Tuple[Union[Type[nn.Module], Callable, MatchAllNode, str], ...], Callable[..., None]] = {(nn.Linear, 'output'): prune_linear, (nn.Linear, nn.Linear): prune_linear_linear, (nn.Conv2d, 'output'): prune_conv2d, (nn.Conv2d, nn.Conv2d): prune_conv2d_conv2d, (nn.LSTM, getitem, nn.Linear): prune_lstm_output_linear, (nn.LSTM, getitem, nn.LayerNorm, nn.Linear): prune_lstm_output_layernorm_linear}
    for activation in chain(_get_supported_activation_functions(), _get_supported_activation_modules()):
        patterns.update({(nn.Linear, activation, nn.Linear): prune_linear_activation_linear, (nn.Conv2d, activation, nn.Conv2d): prune_conv2d_activation_conv2d, (nn.Conv2d, activation, nn.AvgPool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, activation, F.avg_pool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, activation, nn.MaxPool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, activation, F.max_pool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, nn.AvgPool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, F.avg_pool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, nn.MaxPool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, F.max_pool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear): prune_conv2d_pool_flatten_linear, (nn.Conv2d, nn.AdaptiveAvgPool2d, torch.flatten, nn.Linear): prune_conv2d_pool_flatten_linear, (nn.Conv2d, nn.AdaptiveMaxPool2d, nn.Flatten, nn.Linear): prune_conv2d_pool_flatten_linear, (nn.Conv2d, nn.AdaptiveMaxPool2d, torch.flatten, nn.Linear): prune_conv2d_pool_flatten_linear})
    return patterns