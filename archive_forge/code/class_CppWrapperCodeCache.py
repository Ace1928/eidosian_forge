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
class CppWrapperCodeCache:
    cache: Dict[str, CDLL] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code: str, func_name: str, key: str, cuda: bool) -> CDLL:
        name = f'inline_extension_{key}'
        cpp_wrapper_dir = cpp_wrapper_cache_dir(name)
        if not os.path.exists(cpp_wrapper_dir):
            os.makedirs(cpp_wrapper_dir)
        ext = 'so'
        filepath = os.path.join(cpp_wrapper_dir, f'{name}.{ext}')
        log.debug('Cpp wrapper code path %s', filepath)
        if key not in cls.cache:
            log.debug('Cpp wrapper cache miss for %s', filepath)
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                if not os.path.exists(filepath):
                    log.debug('Cpp wrapper building %s', filepath)
                    _cpp_flags = cpp_flags()
                    _opt_flags = optimization_flags()
                    _shared = get_shared()
                    _warning_all_flag = get_warning_all_flag()
                    _ipaths, _lpaths, _libs, _macros, _build_arch_flags = get_include_and_linking_paths(vec_isa=pick_vec_isa(), cuda=cuda)
                    _use_custom_generated_macros = use_custom_generated_macros()
                    _cpp_wrapper_flags = cpp_wrapper_flags()
                    extra_cflags = f'{_cpp_flags} {_opt_flags} {_warning_all_flag} {_build_arch_flags} {_macros}                     {_cpp_wrapper_flags} {_use_custom_generated_macros}'
                    extra_ldflags = f'{_shared} {_lpaths} {_libs} -ffast-math'
                    mod = torch.utils.cpp_extension.load_inline(name=name, build_directory=cpp_wrapper_dir, cpp_sources=[source_code], functions=[func_name], extra_cflags=[extra_cflags], extra_ldflags=[extra_ldflags], extra_include_paths=_ipaths, use_pch=True)
                    log.debug('Cpp wrapper done building %s', filepath)
                else:
                    log.debug('Found target .so, cpp wrapper loading %s', filepath)
                    spec = importlib.util.spec_from_file_location(name, filepath)
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)
                    assert isinstance(spec.loader, abc.Loader)
                    spec.loader.exec_module(mod)
                    log.debug('Cpp wrapper done loading %s', filepath)
                cls.cache[key] = mod
        return cls.cache[key]