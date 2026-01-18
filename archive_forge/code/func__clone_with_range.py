import builtins
import copy
import dataclasses
import inspect
import io
import math
import pathlib
import sys
import typing
from enum import auto, Enum
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch.utils._pytree import (
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
def _clone_with_range(self, lower=2, upper=math.inf):
    from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
    from torch.utils._sympy.value_ranges import ValueRanges
    constraint_range = StrictMinMaxConstraint(vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper), warn_only=False)
    return _create_constraint(self.w_tensor, self.t_id, self.dim, constraint_range, self.shared, self.debug_name)