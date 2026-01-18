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
def _check_abi(self) -> Tuple[str, TorchVersion]:
    if hasattr(self.compiler, 'compiler_cxx'):
        compiler = self.compiler.compiler_cxx[0]
    else:
        compiler = get_cxx_compiler()
    _, version = get_compiler_abi_compatibility_and_version(compiler)
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and ('DISTUTILS_USE_SDK' not in os.environ):
        msg = 'It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.This may lead to multiple activations of the VC env.Please set `DISTUTILS_USE_SDK=1` and try again.'
        raise UserWarning(msg)
    return (compiler, version)