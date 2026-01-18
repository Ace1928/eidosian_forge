import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
import sympy
import torch
from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version
@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """
    architectures: Optional[str] = None
    cuda_version: Optional[str] = None
    operations = 'all'
    build_dir = ''
    curr_build_dir = ''
    generator_target = ''
    kernels = 'all'
    ignore_kernels = ''
    kernel_filter_file = None
    selected_kernel_list = None
    interface_dir = None
    filter_by_cc = True
    disable_full_archs_compilation = False

    def __post_init__(self):
        if self.architectures is None or self.cuda_version is None:
            raise RuntimeError(f'self.architectures={self.architectures!r} or self.cuda_version={self.cuda_version!r} is None!')
        self.architectures = _normalize_cuda_arch(self.architectures)