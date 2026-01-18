import os  # noqa: C101
import sys
from typing import Any, Dict, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
class cpp:
    threads = -1
    no_redundant_loops = True
    dynamic_threads = False
    simdlen = None
    min_chunk_size = 4096
    cxx = (None, os.environ.get('CXX', 'g++'))
    enable_kernel_profile = False
    weight_prepack = True
    inject_relu_bug_TESTING_ONLY = None
    inject_log1p_bug_TESTING_ONLY = None
    vec_isa_ok = None
    descriptive_names = 'original_aten'
    max_horizontal_fusion_size = 16
    fallback_scatter_reduce_sum = True
    enable_unsafe_math_opt_flag = False