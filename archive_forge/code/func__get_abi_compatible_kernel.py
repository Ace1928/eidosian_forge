import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
def _get_abi_compatible_kernel(self):
    if not V.graph.cpp_wrapper:
        return self.kernel

    def sdpa_ver_fn():
        if any((self.get_kwargs_value(arg_name) is None for arg_name in self.ordered_kwargs_for_cpp_kernel)):
            return f'{self.cpp_kernel}_v2'
        else:
            return self.cpp_kernel
    kernel_to_ver = {'at::_scaled_dot_product_flash_attention': sdpa_ver_fn}
    if (ver_fn := kernel_to_ver.get(self.cpp_kernel, None)) is not None:
        return ver_fn()
    return self.cpp_kernel