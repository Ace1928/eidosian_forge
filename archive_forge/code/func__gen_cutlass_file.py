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
def _gen_cutlass_file(file_name: str, cutlass_modules: List[str], src_dir: str, dst_dir: str) -> None:
    orig_full_path = os.path.abspath(os.path.join(src_dir, file_name))
    text = ''
    with open(orig_full_path) as f:
        text = f.read()
    text = _rename_cutlass_import(text, cutlass_modules)
    dst_full_path = os.path.abspath(os.path.join(dst_dir, file_name))
    with open(dst_full_path, 'w') as f:
        f.write(text)