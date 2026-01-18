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
def _get_array_exprs(self, block):
    for instr in block:
        if isinstance(instr, ir.Assign):
            if isinstance(instr.value, ir.Expr):
                if instr.value.op == 'arrayexpr':
                    yield instr