import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class BlockParams:

    def __init__(self, depths: List[int], widths: List[int], group_widths: List[int], bottleneck_multipliers: List[float], strides: List[int], se_ratio: Optional[float]=None) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(cls, depth: int, w_0: int, w_a: float, w_m: float, group_width: int, bottleneck_multiplier: float=1.0, se_ratio: Optional[float]=None, **kwargs: Any) -> 'BlockParams':
        """
        Programmatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """
        QUANT = 8
        STRIDE = 2
        if w_a < 0 or w_0 <= 0 or w_m <= 1 or (w_0 % 8 != 0):
            raise ValueError('Invalid RegNet settings')
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))
        split_helper = zip(block_widths + [0], [0] + block_widths, block_widths + [0], [0] + block_widths)
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]
        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()
        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(stage_widths, bottleneck_multipliers, group_widths)
        return cls(depths=stage_depths, widths=stage_widths, group_widths=group_widths, bottleneck_multipliers=bottleneck_multipliers, strides=strides, se_ratio=se_ratio)

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return (stage_widths, group_widths_min)