from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
class TensorExprTestOptions:

    def __init__(self):
        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        self.old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

    def restore(self):
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        torch._C._get_graph_executor_optimize(self.old_profiling_mode)
        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)