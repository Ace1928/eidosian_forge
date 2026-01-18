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
def _rename_cutlass_import(content: str, cutlass_modules: List[str]) -> str:
    for cutlass_module in cutlass_modules:
        content = content.replace(f'from {cutlass_module} import ', f'from cutlass_library.{cutlass_module} import ')
    return content