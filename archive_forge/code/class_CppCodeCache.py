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
class CppCodeCache:
    cache: Dict[str, CDLL] = dict()
    clear = staticmethod(cache.clear)

    @staticmethod
    def _load_library(path: str) -> CDLL:
        try:
            return cdll.LoadLibrary(path)
        except OSError as e:
            if 'gomp' in str(e) and os.path.exists('/usr/lib64/libgomp.so.1'):
                global _libgomp
                _libgomp = cdll.LoadLibrary('/usr/lib64/libgomp.so.1')
                return cdll.LoadLibrary(path)
            if 'failed to map segment from shared object' in str(e):
                raise OSError(f'{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder is mounted with noexec (e.g., by default Docker mounts tmp file systems as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable.') from e
            raise

    @classmethod
    def load(cls, source_code: str) -> CDLL:
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(cpp_compile_command('i', 'o', vec_isa=picked_vec_isa))
        key, input_path = write(source_code, 'cpp', extra=cpp_command)
        if key not in cls.cache:
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-3] + 'so'
                if not os.path.exists(output_path):
                    cmd = shlex.split(cpp_compile_command(input=input_path, output=output_path, vec_isa=picked_vec_isa))
                    compile_file(input_path, output_path, cmd)
                cls.cache[key] = cls._load_library(output_path)
                cls.cache[key].key = key
        return cls.cache[key]