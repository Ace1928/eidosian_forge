from __future__ import annotations
import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from .._C.libtriton.triton import (ClusterInfo, TMAInfos, add_external_libs, compile_ptx_to_cubin, get_env_vars,
from ..common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from ..common.build import is_hip
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..runtime.jit import (JITFunction, get_cuda_stream, get_current_device, get_device_capability)
from ..tools.disasm import get_sass
from .code_generator import ast_to_ttir
from .make_launcher import make_stub
from .utils import (InfoFromBackendForTensorMap, TensorMapManager, get_ids_of_tensormaps, parse_tma_info)
def _init_handles(self):
    if self.cu_module is not None:
        return
    if self.device_type in ['cuda']:
        device = get_current_device()
        bin_path = {driver.HIP: 'hsaco_path', driver.CUDA: 'cubin'}[driver.backend]
        max_shared = driver.utils.get_device_properties(device)['max_shared_mem']
        fn_load_binary = driver.utils.load_binary
    else:
        assert self.device_backend
        device = self.device_backend.get_current_device()
        bin_path = self.device_backend.get_kernel_bin()
        max_shared = self.device_backend.get_device_properties(device)['max_shared_mem']
        fn_load_binary = self.device_backend.get_load_binary_fn()
    if self.shared > max_shared:
        raise OutOfResources(self.shared, max_shared, 'shared memory')
    mod, func, n_regs, n_spills = fn_load_binary(self.metadata['name'], self.asm[bin_path], self.shared, device)
    self.n_spills = n_spills
    self.n_regs = n_regs
    self.cu_module = mod
    self.cu_function = func