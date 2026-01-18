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
def _jit_compile(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory: str, verbose: bool, with_cuda: Optional[bool], is_python_module, is_standalone, keep_intermediates=True) -> None:
    if is_python_module and is_standalone:
        raise ValueError('`is_python_module` and `is_standalone` are mutually exclusive.')
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any(('cudnn' in f for f in extra_ldflags or []))
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(name, sources, build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths], build_directory=build_directory, with_cuda=with_cuda, is_python_module=is_python_module, is_standalone=is_standalone)
    if version > 0:
        if version != old_version and verbose:
            print(f'The input conditions for extension module {name} have changed. ' + f'Bumping to version {version} and re-building as {name}_v{version}...', file=sys.stderr)
        name = f'{name}_v{version}'
    if version != old_version:
        baton = FileBaton(os.path.join(build_directory, 'lock'))
        if baton.try_acquire():
            try:
                with GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                    if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                        hipify_result = hipify_python.hipify(project_directory=build_directory, output_directory=build_directory, header_include_dirs=extra_include_paths if extra_include_paths is not None else [], extra_files=[os.path.abspath(s) for s in sources], ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')], show_detailed=verbose, show_progress=verbose, is_pytorch_extension=True, clean_ctx=clean_ctx)
                        hipified_sources = set()
                        for source in sources:
                            s_abs = os.path.abspath(source)
                            hipified_sources.add(hipify_result[s_abs].hipified_path if s_abs in hipify_result else s_abs)
                        sources = list(hipified_sources)
                    _write_ninja_file_and_build_library(name=name, sources=sources, extra_cflags=extra_cflags or [], extra_cuda_cflags=extra_cuda_cflags or [], extra_ldflags=extra_ldflags or [], extra_include_paths=extra_include_paths or [], build_directory=build_directory, verbose=verbose, with_cuda=with_cuda, is_standalone=is_standalone)
            finally:
                baton.release()
        else:
            baton.wait()
    elif verbose:
        print(f'No modifications detected for re-loaded extension module {name}, skipping build step...', file=sys.stderr)
    if verbose:
        print(f'Loading extension module {name}...', file=sys.stderr)
    if is_standalone:
        return _get_exec_path(name, build_directory)
    return _import_module_from_library(name, build_directory, is_python_module)