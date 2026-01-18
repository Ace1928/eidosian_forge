import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def _array_expr_to_set(self, expr, out=None):
    """
        Convert an array expression tree into a set of operators.
        """
    if out is None:
        out = set()
    if not isinstance(expr, tuple):
        raise ValueError('{0} not a tuple'.format(expr))
    operation, operands = expr
    processed_operands = []
    for operand in operands:
        if isinstance(operand, tuple):
            operand, _ = self._array_expr_to_set(operand, out)
        processed_operands.append(operand)
    processed_expr = (operation, tuple(processed_operands))
    out.add(processed_expr)
    return (processed_expr, out)