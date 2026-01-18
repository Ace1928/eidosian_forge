import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def _add_alias(lhs, rhs, alias_map, arg_aliases):
    if rhs in arg_aliases:
        arg_aliases.add(lhs)
    else:
        if rhs not in alias_map:
            alias_map[rhs] = set()
        if lhs not in alias_map:
            alias_map[lhs] = set()
        alias_map[rhs].add(lhs)
        alias_map[lhs].add(rhs)
    return