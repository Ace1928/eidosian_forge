from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..ops.misc import Conv2dNormActivation, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class CNBlock(nn.Module):

    def __init__(self, dim, layer_scale: float, stochastic_depth_prob: float, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.block = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True), Permute([0, 2, 3, 1]), norm_layer(dim), nn.Linear(in_features=dim, out_features=4 * dim, bias=True), nn.GELU(), nn.Linear(in_features=4 * dim, out_features=dim, bias=True), Permute([0, 3, 1, 2]))
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result