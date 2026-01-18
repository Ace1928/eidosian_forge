import operator
import os
import sys
from typing import Dict, List, Optional, Tuple
import torch
from lightning_utilities.core.imports import compare_version
from packaging.version import Version
from lightning_fabric.accelerators import XLAAccelerator
from lightning_fabric.accelerators.cuda import num_cuda_devices
from lightning_fabric.accelerators.mps import MPSAccelerator
from lightning_fabric.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
Construct reasons for pytest skipif.

    Args:
        min_cuda_gpus: Require this number of gpus and that the ``PL_RUN_CUDA_TESTS=1`` environment variable is set.
        min_torch: Require that PyTorch is greater or equal than this version.
        max_torch: Require that PyTorch is less than this version.
        min_python: Require that Python is greater or equal than this version.
        bf16_cuda: Require that CUDA device supports bf16.
        tpu: Require that TPU is available.
        mps: If True: Require that MPS (Apple Silicon) is available,
            if False: Explicitly Require that MPS is not available
        skip_windows: Skip for Windows platform.
        standalone: Mark the test as standalone, our CI will run it in a separate process.
            This requires that the ``PL_RUN_STANDALONE_TESTS=1`` environment variable is set.
        deepspeed: Require that microsoft/DeepSpeed is installed.
        dynamo: Require that `torch.dynamo` is supported.

    