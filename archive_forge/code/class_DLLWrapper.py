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
class DLLWrapper:
    """A wrapper for a dynamic library."""

    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self.DLL = cdll.LoadLibrary(lib_path)
        self.is_open = True

    def close(self):
        if self.is_open:
            self._dlclose()
            self.is_open = False

    def _dlclose(self):
        f_dlclose = None
        if is_linux():
            syms = CDLL(None)
            if not hasattr(syms, 'dlclose'):
                syms = CDLL('libc.so')
            if hasattr(syms, 'dlclose'):
                f_dlclose = syms.dlclose
        else:
            raise NotImplementedError('Unsupported env, failed to do dlclose!')
        if f_dlclose is not None:
            f_dlclose.argtypes = [c_void_p]
            f_dlclose(self.DLL._handle)
        else:
            log.warning('dll unloading function was not found, library may not be unloaded properly!')

    def __getattr__(self, name):
        if not self.is_open:
            raise RuntimeError(f'Cannot use closed DLL library: {self.lib_path}')
        method = getattr(self.DLL, name)

        def _wrapped_func(*args):
            err = method(*args)
            if err:
                raise RuntimeError(f'Error in function: {method.__name__}')
        return _wrapped_func

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()