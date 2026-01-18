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
def _get_stack_summary(self):
    fsummary = None
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code.co_filename not in uninteresting_files():
                fsummary = traceback.FrameSummary(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                break
            frame = frame.f_back
    finally:
        del frame
    maybe_user_loc = ''
    user_tb = TracingContext.extract_stack()
    if user_tb:
        maybe_user_loc = ' at ' + format_frame(user_tb[-1])
    return (fsummary, user_tb, maybe_user_loc)