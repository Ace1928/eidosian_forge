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
def _get_pybind11_abi_build_flags():
    abi_cflags = []
    for pname in ['COMPILER_TYPE', 'STDLIB', 'BUILD_ABI']:
        pval = getattr(torch._C, f'_PYBIND11_{pname}')
        if pval is not None and (not IS_WINDOWS):
            abi_cflags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')
    return abi_cflags