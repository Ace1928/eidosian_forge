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