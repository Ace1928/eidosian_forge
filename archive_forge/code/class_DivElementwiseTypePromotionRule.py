from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    """Reference type promotion rule from torch._refs.div.

    Rule depends on the value of the `rounding_mode` argument.
    """

    def __init__(self):
        super().__init__('aten', 'div', promote_args_positions=(0, 1), promote_kwargs_names=(), promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

    def preview_type_promotion(self, args: tuple, kwargs: dict) -> TypePromotionSnapshot:
        rounding_mode = kwargs.get('rounding_mode', None)
        if rounding_mode is None:
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == 'trunc':
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == 'floor':
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        raise ValueError(f'Unknown rounding_mode: {rounding_mode}')