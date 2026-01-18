import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
@dataclass(frozen=True)
class StrictMinMaxConstraint(Constraint):
    """
    For clients: the size at this dimension must be within 'vr' (which
    specifies a lower and upper bound, inclusive-inclusive) AND it
    must be non-negative and should not be 0 or 1 (but see NB below).

    For backends: there must not be any guards on this dimension which
    are not implied by the given lower and upper bound.  Regardless of
    the lower bound, the backend can assume the size is non-negative
    and that it is not 0 or 1.

    An unbounded StrictMinMaxConstraint can be thought of as a strict version
    of "RelaxedUnspecConstraint".

    NB: Export will often unsoundly assume that a graph works for 0/1, even
    though at trace time we assumed size is not 0 or 1.  The idea is that
    if we produce a graph that works for a range of values, it will be OK
    for N=0/1 too.
    """
    vr: ValueRanges

    def render(self, source: Source):
        return f'{self.vr.lower} <= {source.name()} <= {self.vr.upper}'