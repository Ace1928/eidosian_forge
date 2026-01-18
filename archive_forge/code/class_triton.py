import os  # noqa: C101
import sys
from typing import Any, Dict, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
class triton:
    cudagraphs = False
    cudagraph_trees = True
    slow_path_cudagraph_asserts = True
    cudagraph_trees_history_recording = False
    fast_path_cudagraph_asserts = False
    skip_cudagraph_warmup = False
    debug_sync_graph = False
    debug_sync_kernel = False
    dense_indexing = False
    max_tiles = 2
    autotune_pointwise = True
    autotune_cublasLt = True
    tiling_prevents_pointwise_fusion = True
    tiling_prevents_reduction_fusion = True
    unique_kernel_names = os.environ.get('TORCHINDUCTOR_UNIQUE_KERNEL_NAMES') == '1'
    descriptive_names = 'original_aten'
    persistent_reductions = os.environ.get('TORCHINDUCTOR_PERSISTENT_REDUCTIONS', '1') == '1'
    divisible_by_16 = True
    max_block = {'X': 2048, 'Y': 1024, 'Z': 1024}
    store_cubin = False
    spill_threshold: int = 16
    inject_relu_bug_TESTING_ONLY = None