from __future__ import annotations
import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import torch
from torch._dynamo.device_interface import (
from torch._dynamo.utils import counters
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
from torch.hub import _Faketqdm, tqdm
import torch
from ctypes import cdll
def _nvcc_compiler_options() -> List[str]:
    arch = cuda_env.get_cuda_arch()
    if arch == '90':
        arch = '90a'
    code = [f'sm_{arch}', f'compute_{arch}']
    if config.cuda.enable_cuda_lto:
        code += [f'lto_{arch}']
    options = ['-t=0', '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1', '-w', f'-gencode=arch=compute_{arch},code=[{','.join(code)}]', config.cuda.compile_opt_level, '-std=c++17', '--expt-relaxed-constexpr']
    if config.cuda.enable_debug_info:
        options.extend(['-lineinfo', '-g', '-DCUTLASS_DEBUG_TRACE_LEVEL=1'])
    if config.cuda.enable_ptxas_info:
        options.extend(['--keep', '--ptxas-options=--warn-on-local-memory-usage', '--ptxas-options=--warn-on-spills', '--resource-usage', '--source-in-ptx'])
    if config.cuda.use_fast_math:
        options.extend(['--use_fast_math', '-DCUTLASS_USE_TANH_FOR_SIGMOID=1'])
    return options