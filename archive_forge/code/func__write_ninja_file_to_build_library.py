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
def _write_ninja_file_to_build_library(path, name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, with_cuda, is_standalone) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]
    user_includes = [os.path.abspath(file) for file in extra_include_paths]
    system_includes = include_paths(with_cuda)
    python_include_path = sysconfig.get_path('include', scheme='nt' if IS_WINDOWS else 'posix_prefix')
    if python_include_path is not None:
        system_includes.append(python_include_path)
    if IS_WINDOWS:
        user_includes += system_includes
        system_includes.clear()
    common_cflags = []
    if not is_standalone:
        common_cflags.append(f'-DTORCH_EXTENSION_NAME={name}')
        common_cflags.append('-DTORCH_API_INCLUDE_EXTENSION_H')
    common_cflags += [f'{x}' for x in _get_pybind11_abi_build_flags()]
    common_cflags += [f'-I{include}' for include in user_includes]
    common_cflags += [f'-isystem {include}' for include in system_includes]
    common_cflags += [f'{x}' for x in _get_glibcxx_abi_build_flags()]
    if IS_WINDOWS:
        cflags = common_cflags + COMMON_MSVC_FLAGS + ['/std:c++17'] + extra_cflags
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ['-fPIC', '-std=c++17'] + extra_cflags
    if with_cuda and IS_HIP_EXTENSION:
        cuda_flags = ['-DWITH_HIP'] + cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS
        cuda_flags += extra_cuda_cflags
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
    elif with_cuda:
        cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags()
        if IS_WINDOWS:
            for flag in COMMON_MSVC_FLAGS:
                cuda_flags = ['-Xcompiler', flag] + cuda_flags
            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                cuda_flags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cuda_flags
            cuda_flags = cuda_flags + ['-std=c++17']
            cuda_flags = _nt_quote_args(cuda_flags)
            cuda_flags += _nt_quote_args(extra_cuda_cflags)
        else:
            cuda_flags += ['--compiler-options', "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            if not any((flag.startswith('-std=') for flag in cuda_flags)):
                cuda_flags.append('-std=c++17')
            cc_env = os.getenv('CC')
            if cc_env is not None:
                cuda_flags = ['-ccbin', cc_env] + cuda_flags
    else:
        cuda_flags = None

    def object_file_path(source_file: str) -> str:
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            target = f'{file_name}.cuda.o'
        else:
            target = f'{file_name}.o'
        return target
    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags
    if IS_MACOS:
        ldflags.append('-undefined dynamic_lookup')
    elif IS_WINDOWS:
        ldflags = _nt_quote_args(ldflags)
    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f'{name}{ext}'
    _write_ninja_file(path=path, cflags=cflags, post_cflags=None, cuda_cflags=cuda_flags, cuda_post_cflags=None, cuda_dlink_post_cflags=None, sources=sources, objects=objects, ldflags=ldflags, library_target=library_target, with_cuda=with_cuda)