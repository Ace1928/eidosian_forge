import os  # noqa: C101
import sys
from typing import Any, Dict, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
class cuda:
    arch = None
    version = None
    compile_opt_level = '-O1'
    enable_cuda_lto = False
    enable_ptxas_info = False
    enable_debug_info = False
    use_fast_math = False
    cutlass_dir = os.environ.get('TORCHINDUCTOR_CUTLASS_DIR', os.path.abspath(os.path.join(os.path.dirname(torch.__file__), '../third_party/cutlass/')))
    cutlass_max_profiling_configs = None
    cuda_cxx = None
    cutlass_only_evt_capable_ops: bool = False