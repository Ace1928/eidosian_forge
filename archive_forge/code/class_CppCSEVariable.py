import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppCSEVariable(CSEVariable):

    def __init__(self, name, bounds: ValueRanges):
        super().__init__(name, bounds)
        self.is_vec = False
        self.dtype: Optional[torch.dtype] = None
        self.dependent_itervars: Set[sympy.Symbol] = set()

    def update_on_args(self, name, args, kwargs):
        if name == 'load':
            self._set_dependent_itervars(args[1])
        else:
            self.dependent_itervars.update(*[arg.dependent_itervars for arg in args if isinstance(arg, CppCSEVariable)])
            if name == 'index_expr':
                self._set_dependent_itervars(args[0])
            if any((arg.is_vec for arg in args if isinstance(arg, CppCSEVariable))):
                self.is_vec = True
        if hasattr(V.interpreter, 'current_node') and get_current_node_opt_ctx() is not None:
            self.dtype = get_current_node_opt_ctx().dtype

    def _set_dependent_itervars(self, index: sympy.Expr):
        """
        Set the relevant itervars for this variable based on the `index` expression.
        This includes the itervars directly used in the `index` as well as relevant itervars
        of other cse variables used in the `index`.
        """
        for s in index.free_symbols:
            if s in V.kernel.itervars:
                self.dependent_itervars.add(s)
            elif s.name in V.kernel.cse.varname_map:
                self.dependent_itervars.update(V.kernel.cse.varname_map[s.name].dependent_itervars)

    def depends_on(self, itervar: sympy.Symbol):
        return itervar in self.dependent_itervars