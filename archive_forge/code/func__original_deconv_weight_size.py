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
def _original_deconv_weight_size(prepacked_weight, groups):
    prepacked_weight_size = prepacked_weight.size()
    dim = len(prepacked_weight_size)
    assert dim > 2, 'Expect weight dim > 2'
    if groups > 1:
        weight_size = []
        weight_size.append(prepacked_weight_size[1] * groups)
        weight_size.append(prepacked_weight_size[0] / groups)
        for d in range(2, dim):
            weight_size.append(prepacked_weight_size[d])
    else:
        weight_size = prepacked_weight.transpose(0, 1).size()
    return weight_size