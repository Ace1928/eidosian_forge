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
class ReductionTypePromotionRule(TypePromotionRule):

    def __init__(self, namespace: str, op_name: str, promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND):
        """Constructs a TypePromotionRule for reduction operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.sum'.
            op_name: Name of the op. E.g. 'sum' in 'torch.ops.aten.sum'.
            promotion_kind: Type promotion kind. Refer to [_prims_common.reduction_dtypes]((https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py)) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promotion_kind = promotion_kind

    def __repr__(self):
        return f"ReductionTypePromotionRule('{self.namespace}', '{self.op_name}', {self.promotion_kind})"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ElementwiseTypePromotionRule):
            return False
        return self.namespace == __value.namespace and self.op_name == __value.op_name and (self.promotion_kind == __value.promotion_kind)

    def __hash__(self) -> int:
        return f'{type(self)}:{self.namespace}.{self.op_name}'.__hash__()

    def preview_type_promotion(self, args: tuple, kwargs: dict) -> TypePromotionSnapshot:
        assert len(args) >= 1 and isinstance((arg := args[0]), torch.Tensor), f'Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument'
        dtype: Optional[torch.dtype] = kwargs.get('dtype', None)
        computation_dtype, result_dtype = _prims_common.reduction_dtypes(arg, self.promotion_kind, dtype)
        if result_dtype is None:
            result_dtype = computation_dtype
        return TypePromotionSnapshot({0: computation_dtype}, {}, result_dtype)