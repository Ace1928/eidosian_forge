import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
@functools.lru_cache(None)
def _cuda_system_info_comment():
    if not torch.cuda.is_available():
        return '# torch.cuda.is_available()==False, no GPU info collected\n'
    model_str = '# CUDA Info: \n'
    try:
        cuda_version_out = subprocess.check_output(['nvcc', '--version'])
        cuda_version_lines = cuda_version_out.decode().split('\n')
        comment = ''.join([f'# {s} \n' for s in cuda_version_lines if s not in ['']])
        model_str += f'{comment}\n'
    except FileNotFoundError:
        model_str += '# nvcc not found\n'
    gpu_names = Counter((torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
    model_str += '# GPU Hardware Info: \n'
    for name, count in gpu_names.items():
        model_str += f'# {name} : {count} \n'
    model_str += '\n'
    return model_str