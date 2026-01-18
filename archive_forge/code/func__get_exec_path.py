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
def _get_exec_path(module_name, path):
    if IS_WINDOWS and TORCH_LIB_PATH not in os.getenv('PATH', '').split(';'):
        torch_lib_in_path = any((os.path.exists(p) and os.path.samefile(p, TORCH_LIB_PATH) for p in os.getenv('PATH', '').split(';')))
        if not torch_lib_in_path:
            os.environ['PATH'] = f'{TORCH_LIB_PATH};{os.getenv('PATH', '')}'
    return os.path.join(path, f'{module_name}{EXEC_EXT}')