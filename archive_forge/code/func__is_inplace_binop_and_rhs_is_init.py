import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def _is_inplace_binop_and_rhs_is_init(inst, redvar_name):
    """Is ``inst`` an inplace-binop and the RHS is the reduction init?
    """
    if not isinstance(inst, ir.Assign):
        return False
    rhs = inst.value
    if not isinstance(rhs, ir.Expr):
        return False
    if rhs.op != 'inplace_binop':
        return False
    if rhs.rhs.name != f'{redvar_name}#init':
        return False
    return True