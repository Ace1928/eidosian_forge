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
class CUDACodeCache:

    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str
    cache: Dict[str, CacheEntry] = dict()
    clear = staticmethod(cache.clear)
    _SOURCE_CODE_SUFFIX = 'cu'

    @classmethod
    def write(cls, source_code, dst_file_ext) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """
        cuda_command = repr(cuda_compile_command(['dummy_input'], 'dummy_output', dst_file_ext))
        key, input_path = write(source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command)
        return (key, input_path)

    @classmethod
    def compile(cls, source_code, dst_file_ext) -> Tuple[str, str, str]:
        """
        Compiles CUDA source_code into a file with dst_file_ext extension.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        key, input_path = cls.write(source_code, dst_file_ext)
        if key not in cls.cache:
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                output_path = input_path[:-len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    cmd = cuda_compile_command([input_path], output_path, dst_file_ext).split(' ')
                    try:
                        subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=os.environ)
                    except subprocess.CalledProcessError as error:
                        raise exc.CUDACompileError(cmd, error.output) from error
                cls.cache[key] = CUDACodeCache.CacheEntry(input_path, output_path)
        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code, dst_file_ext) -> Tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """
        if dst_file_ext != 'so':
            raise RuntimeError(f'Only support loading a .so file for now. Requested file extension: {dst_file_ext}. Source code: {source_code}')
        dst_file_path, hash_key, source_code_path = cls.compile(source_code, dst_file_ext)
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)