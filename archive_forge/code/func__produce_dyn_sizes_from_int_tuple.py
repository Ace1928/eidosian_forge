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
def _produce_dyn_sizes_from_int_tuple(self, tensor_size: Tuple[int], source: Source, symbolic_context: SymbolicContext) -> List[sympy.Expr]:
    assert all((not is_symbolic(val) for val in tensor_size)), f'Expect size to be a plain tuple of ints but got {tensor_size}'
    from torch._dynamo.source import TensorPropertySource, TensorProperty
    _assert_symbol_context(symbolic_context)
    dynamic_dims = symbolic_context.dynamic_sizes
    constraint_dims = symbolic_context.constraint_sizes
    size = []
    for i, val in enumerate(tensor_size):
        size.append(self.create_symbol(val, TensorPropertySource(source, TensorProperty.SIZE, i), dynamic_dims[i], constraint_dims[i]))
    return size