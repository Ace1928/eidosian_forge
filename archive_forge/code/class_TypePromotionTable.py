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
class TypePromotionTable:
    """Type promotion table for torch.ops."""

    def __init__(self):
        self._rule_table = {}
        for rule in _GENERATED_ATEN_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)
        for rule in _EXTRA_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)

    @_beartype.beartype
    def add_rule(self, rule: TypePromotionRule) -> None:
        """Add a type promotion rule for a python op in a torch.ops module.

        Args:
            rule: Type promotion rule.
            module: Module containing the op. E.g. torch.ops.aten.

        Raises:
            ValueError: If the rule is invalid.
        """
        if not rule.is_valid():
            raise ValueError(f'Invalid type promotion rule: {rule}')
        self._rule_table[f'{rule.namespace}.{rule.op_name}'] = rule

    @_beartype.beartype
    def get_rule(self, py_op: torch._ops.OpOverloadPacket) -> Optional[TypePromotionRule]:
        """Get type promotion rule for a python op under 'torch.ops.<namespace>'."""
        return self._rule_table.get(str(py_op), None)