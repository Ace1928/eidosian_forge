from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import (
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
class FailedMatch(RuntimeError):

    def __init__(self, format_string, *args, **kwargs):
        self.format_string = format_string
        if len(format_string) > 200:
            raise RuntimeError(f'Format string too long - use lazy construction of strings instead. Format string is\n {format_string}')
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self):
        return False