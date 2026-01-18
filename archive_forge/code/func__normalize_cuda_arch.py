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
def _normalize_cuda_arch(arch: str) -> str:
    if int(arch) >= 90:
        return '90'
    elif int(arch) >= 80:
        return '80'
    elif int(arch) >= 75:
        return '75'
    elif int(arch) >= 70:
        return '70'
    else:
        raise NotImplementedError(f'Unsupported cuda arch: {arch}')