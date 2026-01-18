import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
class ExternKernelCaller(ChoiceCaller):

    def __init__(self, choice: ExternKernelChoice, input_nodes, layout, kwargs=None, *, has_out_variant=True):
        super().__init__(choice.name, input_nodes, layout)
        self.choice = choice
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def __str__(self):
        return f'ExternKernelCaller({self.choice.call_name()})'

    def benchmark(self, *args, out):
        if self.has_out_variant:
            return super().benchmark(*args, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*args)
            torch._C._dynamo.guards.assert_size_stride(out_new, tuple(out.size()), tuple(out.stride()))
            out.copy_(out_new)
            return do_bench(lambda: algo(*args))

    def to_callable(self):
        fn = self.choice.to_callable()
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)
        else:
            return fn

    def hash_key(self):
        return '-'.join([self.choice.name, *[f'{kwarg}={repr(self.kwargs[kwarg])}' for kwarg in sorted(self.kwargs.keys())], self.choice.hash_key()])

    def output_node(self):
        cls: Union[Type[ir.ExternKernelOut], Type[ir.ExternKernelAlloc]]
        if self.has_out_variant:
            cls = ir.ExternKernelOut
        else:
            cls = ir.ExternKernelAlloc
        return ir.TensorBox.create(cls(layout=self.layout, inputs=self.input_nodes, kernel=self.choice.call_name(), cpp_kernel=self.choice.cpp_kernel, ordered_kwargs_for_cpp_kernel=self.choice.ordered_kwargs_for_cpp_kernel, kwargs=self.kwargs))