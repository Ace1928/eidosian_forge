import math
import os
import sys
import platform
import textwrap
import ctypes
import inspect
from ._utils import _import_dotted_name, classproperty
from ._utils import _functionalize_sync as _sync
from ._utils_internal import get_file_path, prepare_multiprocessing_environment, \
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING, Union, List
import builtins
from math import e , nan , inf , pi
from ._tensor import Tensor
from .storage import _StorageBase, TypedStorage, _LegacyStorage, UntypedStorage, _warn_typed_storage_removal
from .random import set_rng_state, get_rng_state, manual_seed, initial_seed, seed
from .serialization import save, load
from ._tensor_str import set_printoptions
from torch.amp import autocast
from ._compile import _disable_dynamo
from .functional import *  # noqa: F403
from torch import cuda as cuda
from torch import cpu as cpu
from torch import mps as mps
from torch import autograd as autograd
from torch.autograd import (
from torch import fft as fft
from torch import futures as futures
from torch import _awaits as _awaits
from torch import nested as nested
from torch import nn as nn
from torch.signal import windows as windows
from torch import optim as optim
import torch.optim._multi_tensor
from torch import multiprocessing as multiprocessing
from torch import sparse as sparse
from torch import special as special
import torch.utils.backcompat
from torch import jit as jit
from torch import linalg as linalg
from torch import hub as hub
from torch import random as random
from torch import distributions as distributions
from torch import testing as testing
from torch import backends as backends
import torch.utils.data
from torch import __config__ as __config__
from torch import __future__ as __future__
from torch import profiler as profiler
from torch import ao as ao
import torch.nn.quantizable
import torch.nn.quantized
import torch.nn.qat
import torch.nn.intrinsic
from . import _torch_docs, _tensor_docs, _storage_docs
from torch._ops import ops
from torch._classes import classes
import torch._library
from torch import quantization as quantization
from torch import quasirandom as quasirandom
from torch.multiprocessing._atfork import register_after_fork
from ._lobpcg import lobpcg as lobpcg
from torch.utils.dlpack import from_dlpack, to_dlpack
from . import masked
from ._linalg_utils import (  # type: ignore[misc]
from ._linalg_utils import _symeig as symeig  # type: ignore[misc]
from torch import export as export
from torch._higher_order_ops import cond
from . import return_types
from . import library
import torch.fx.experimental.sym_node
from torch import func as func
from torch.func import vmap
from . import _logging
class _TorchCompileInductorWrapper:
    compiler_name = 'inductor'

    def __init__(self, mode, options, dynamic):
        self.config: Dict[str, Any] = dict()
        self.dynamic = dynamic
        self.apply_mode(mode)
        self.apply_options(options)
        if self.config.get('triton.cudagraphs', False):
            os.environ['DISABLE_CUPTI_LAZY_REINIT'] = '1'
            os.environ['TEARDOWN_CUPTI'] = '0'

    def __eq__(self, other):
        return isinstance(other, _TorchCompileInductorWrapper) and self.config == other.config and (self.dynamic == other.dynamic)

    def apply_mode(self, mode: Optional[str]):
        if mode is None or mode == 'default':
            pass
        elif mode in ('reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'):
            from torch._inductor import list_mode_options
            self.apply_options(list_mode_options(mode, self.dynamic))
        else:
            raise RuntimeError(f'Unrecognized mode={mode}, should be one of: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs')

    def apply_options(self, options: Optional[Dict[str, Any]]):
        if not options:
            return
        from torch._inductor import config
        current_config: Dict[str, Any] = config.shallow_copy_dict()
        for key, val in options.items():
            attr_name = key.replace('-', '_')
            if attr_name not in current_config:
                raise RuntimeError(f'Unexpected optimization option {key}, known options are {list(current_config.keys())}')
            if type(val) is not type(current_config[attr_name]):
                val_type_str = type(val).__name__
                expected_type_str = type(current_config[attr_name]).__name__
                raise RuntimeError(f'Unexpected type of attr {key}, got {val_type_str} should be {expected_type_str}')
            self.config[attr_name] = val

    def __call__(self, model_, inputs_):
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(model_, inputs_, config_patches=self.config)

    def get_compiler_config(self):
        from torch._inductor.compile_fx import get_patched_config_dict
        return get_patched_config_dict(config_patches=self.config)

    def reset(self):
        from torch._inductor import config
        if 'triton.cudagraphs' in self.config or config.triton.cudagraphs:
            if self.config.get('triton.cudagraphs', True):
                from torch._inductor.cudagraph_trees import reset_cudagraph_trees
                reset_cudagraph_trees()