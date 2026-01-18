import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import collections
from pathlib import Path
import errno
import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from .hipify import hipify_python
from .hipify.hipify_python import GeneratedFileCleaner
from typing import Dict, List, Optional, Union, Tuple
from torch.torch_version import TorchVersion, Version
from setuptools.command.build_ext import build_ext
def _get_cuda_arch_flags(cflags: Optional[List[str]]=None) -> List[str]:
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    if cflags is not None:
        for flag in cflags:
            if 'TORCH_EXTENSION_NAME' in flag:
                continue
            if 'arch' in flag:
                return []
    named_arches = collections.OrderedDict([('Kepler+Tesla', '3.7'), ('Kepler', '3.5+PTX'), ('Maxwell+Tegra', '5.3'), ('Maxwell', '5.0;5.2+PTX'), ('Pascal', '6.0;6.1+PTX'), ('Volta+Tegra', '7.2'), ('Volta', '7.0+PTX'), ('Turing', '7.5+PTX'), ('Ampere+Tegra', '8.7'), ('Ampere', '8.0;8.6+PTX'), ('Ada', '8.9+PTX'), ('Hopper', '9.0+PTX')])
    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2', '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a']
    valid_arch_strings = supported_arches + [s + '+PTX' for s in supported_arches]
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if not _arch_list:
        arch_list = []
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int(arch.split('_')[1]) for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            max_supported_sm = max(((sm // 10, sm % 10) for sm in supported_sm))
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
        arch_list[-1] += '+PTX'
    else:
        _arch_list = _arch_list.replace(' ', ';')
        for named_arch, archval in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archval)
        arch_list = _arch_list.split(';')
    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f'Unknown CUDA arch ({arch}) or GPU not supported')
        else:
            num = arch[0] + arch[2:].split('+')[0]
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')
    return sorted(set(flags))