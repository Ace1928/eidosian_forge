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
def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils
        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]
        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fileno = 1
        subprocess.run(command, stdout=stdout_fileno if verbose else subprocess.PIPE, stderr=subprocess.STDOUT, cwd=build_directory, check=True, env=env)
    except subprocess.CalledProcessError as e:
        _, error, _ = sys.exc_info()
        message = error_prefix
        if hasattr(error, 'output') and error.output:
            message += f': {error.output.decode(*SUBPROCESS_DECODE_ARGS)}'
        raise RuntimeError(message) from e