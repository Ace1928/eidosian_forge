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
class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self):
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: Dict[PatternExpr, str] = {}
        self.memoized_objs_pp: Dict[PatternExpr, str] = {}

    @staticmethod
    def run(obj: PatternExpr, output_name='output'):
        """
        Serializes obj to python code with obj written out to `output_name`
        """
        pp = PatternPrettyPrinter()
        assert hasattr(obj, 'pretty_print')
        out_str = obj.pretty_print(pp=pp)
        output = []
        for key in pp.memoized_objs_names:
            output.append(f'{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}')
        output.append(f'{output_name} = {out_str}')
        return '\n'.join(output)

    def pretty_print(self, obj):
        if isinstance(obj, _TargetArgsExpr):
            if (memoized_name := self.memoized_objs_names.get(obj)):
                return memoized_name
            else:
                return self.memoize(obj)
        if hasattr(obj, 'pretty_print'):
            return obj.pretty_print(self)
        return repr(obj)

    def memoize(self, obj):
        obj_str = obj.pretty_print(self)
        obj_name = obj.fns_repr()
        for prefix in ('aten.', 'torch.', 'prims.'):
            obj_name = obj_name.replace(prefix, '')
        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        return tmp_name