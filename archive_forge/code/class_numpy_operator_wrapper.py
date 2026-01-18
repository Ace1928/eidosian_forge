import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
class numpy_operator_wrapper:
    """Implements dunder methods for tnp.ndarray via functions from the operator library"""

    def __init__(self, op: Callable[..., Any]):
        self.op = op
        self.__name__ = f'wrapped_{op.__name__}'

    def __repr__(self):
        return f'<Wrapped operator <original {self.__name__}>>'

    def __call__(self, *args, **kwargs):
        assert not kwargs
        args = (tnp.ndarray(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
        out = self.op(*args)
        return numpy_to_tensor(out)