import contextlib
import copy
import functools
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import (
import srsly
from .backends import CupyOps, NumpyOps, Ops, ParamServer, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .shims import Shim
from .types import FloatsXd
from .util import (
def _to_ops(self, ops: Ops) -> None:
    """Common method for to_cpu/to_gpu."""
    for node in self.walk():
        node.ops = ops
        for name in node.param_names:
            if node.has_param(name):
                node.set_param(name, ops.asarray_f(node.get_param(name)))
            if node.has_grad(name):
                node.set_grad(name, ops.asarray_f(node.get_grad(name)))
        for shim in node.shims:
            shim.to_device(ops.device_type, ops.device_id)