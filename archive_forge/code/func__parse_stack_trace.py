import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def _parse_stack_trace(stack_trace: str):
    if stack_trace is None:
        return None
    ParsedStackTrace = collections.namedtuple('ParsedStackTrace', ['file', 'lineno', 'code'])
    pattern = re.compile('^File \\"(.+)\\", line (\\d+), in (.+)$')
    lines = stack_trace.strip().split('\n')
    summary_str = ''
    for idx in range(len(lines) - 2, -1, -1):
        line = lines[idx].strip()
        matches = pattern.match(line)
        if matches:
            file = matches.group(1)
            lineno = matches.group(2)
            code = lines[idx + 1].strip()
            return ParsedStackTrace(file, lineno, code)
    return None