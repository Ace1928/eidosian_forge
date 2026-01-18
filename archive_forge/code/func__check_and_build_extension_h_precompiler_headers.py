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
def _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths, is_standalone=False):
    """
    Precompiled Headers(PCH) can pre-build the same headers and reduce build time for pytorch load_inline modules.
    GCC offical manual: https://gcc.gnu.org/onlinedocs/gcc-4.0.4/gcc/Precompiled-Headers.html
    PCH only works when built pch file(header.h.gch) and build target have the same build parameters. So, We need
    add a signature file to record PCH file parameters. If the build parameters(signature) changed, it should rebuild
    PCH file.

    Note:
    1. Windows and MacOS have different PCH mechanism. We only support Linux currently.
    2. It only works on GCC/G++.
    """
    if not IS_LINUX:
        return
    compiler = get_cxx_compiler()
    b_is_gcc = check_compiler_is_gcc(compiler)
    if b_is_gcc is False:
        return
    head_file = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h')
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    def listToString(s):
        string = ''
        if s is None:
            return string
        for element in s:
            string += element + ' '
        return string

    def format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags, torch_include_dirs, extra_cflags, extra_include_paths):
        return re.sub('[ \\n]+', ' ', f'\n                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}\n            ').strip()

    def command_to_signature(cmd):
        signature = cmd.replace(' ', '_')
        return signature

    def check_pch_signature_in_file(file_path, signature):
        b_exist = os.path.isfile(file_path)
        if b_exist is False:
            return False
        with open(file_path) as file:
            content = file.read()
            if signature == content:
                return True
            else:
                return False

    def _create_if_not_exist(path_dir):
        if not os.path.exists(path_dir):
            try:
                Path(path_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise RuntimeError(f'Fail to create path {path_dir}') from exc

    def write_pch_signature_to_file(file_path, pch_sign):
        _create_if_not_exist(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            f.write(pch_sign)
            f.close()

    def build_precompile_header(pch_cmd):
        try:
            subprocess.check_output(pch_cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Compile PreCompile Header fail, command: {pch_cmd}') from e
    extra_cflags_str = listToString(extra_cflags)
    extra_include_paths_str = ' '.join([f'-I{include}' for include in extra_include_paths] if extra_include_paths else [])
    lib_include = os.path.join(_TORCH_PATH, 'include')
    torch_include_dirs = [f'-I {lib_include}', '-I {}'.format(sysconfig.get_path('include')), '-I {}'.format(os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'))]
    torch_include_dirs_str = listToString(torch_include_dirs)
    common_cflags = []
    if not is_standalone:
        common_cflags += ['-DTORCH_API_INCLUDE_EXTENSION_H']
    common_cflags += ['-std=c++17', '-fPIC']
    common_cflags += [f'{x}' for x in _get_pybind11_abi_build_flags()]
    common_cflags += [f'{x}' for x in _get_glibcxx_abi_build_flags()]
    common_cflags_str = listToString(common_cflags)
    pch_cmd = format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags_str, torch_include_dirs_str, extra_cflags_str, extra_include_paths_str)
    pch_sign = command_to_signature(pch_cmd)
    if os.path.isfile(head_file_pch) is not True:
        build_precompile_header(pch_cmd)
        write_pch_signature_to_file(head_file_signature, pch_sign)
    else:
        b_same_sign = check_pch_signature_in_file(head_file_signature, pch_sign)
        if b_same_sign is False:
            build_precompile_header(pch_cmd)
            write_pch_signature_to_file(head_file_signature, pch_sign)