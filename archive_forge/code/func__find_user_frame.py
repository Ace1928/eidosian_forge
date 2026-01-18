import enum
import dis
import copy
import sys
import torch
import inspect
import operator
import traceback
import collections
from dataclasses import is_dataclass, fields
from .graph import magic_methods, reflectable_magic_methods, Graph
from typing import Tuple, Dict, OrderedDict, Optional, Any, Iterator, Callable
from .node import Target, Node, Argument, base_types, map_aggregate
from ._compatibility import compatibility
from .operator_schemas import check_for_mutable_operation
import torch.fx.traceback as fx_traceback
def _find_user_frame(self):
    """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
    frame = inspect.currentframe()
    pt_files = ['torch/fx/proxy.py', 'torch/fx/_symbolic_trace.py', 'torch/fx/experimental/proxy_tensor.py', 'torch/_ops.py', 'torch/_tensor.py', 'torch/utils/_python_dispatch.py', 'torch/_prims_common/wrappers.py', 'torch/_refs/__init__.py', 'torch/_refs/nn/functional/__init__.py', 'torch/utils/_stats.py']
    while frame:
        frame = frame.f_back
        if frame and all((not frame.f_code.co_filename.endswith(file) for file in pt_files)):
            break
    if not frame:
        return None
    return frame