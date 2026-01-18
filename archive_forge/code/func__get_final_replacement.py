import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
def _get_final_replacement(self, replacement_map, instr):
    """Find the final replacement instruction for a given initial
        instruction by chasing instructions in a map from instructions
        to replacement instructions.
        """
    replacement = replacement_map[instr]
    while replacement in replacement_map:
        replacement = replacement_map[replacement]
    return replacement