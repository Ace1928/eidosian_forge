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
class CacheBase:

    @staticmethod
    @functools.lru_cache(None)
    def get_system() -> Dict[str, Any]:
        try:
            import triton
            triton_version = triton.__version__
        except ModuleNotFoundError:
            triton_version = None
        try:
            system: Dict[str, Any] = {'device': {'name': torch.cuda.get_device_properties(torch.cuda.current_device()).name}, 'version': {'cuda': torch.version.cuda, 'triton': triton_version}, 'other': {'allow_tf32': torch.backends.cuda.matmul.allow_tf32}}
        except (AssertionError, RuntimeError):
            system = {}
        system['hash'] = hashlib.sha256(json.dumps(system, sort_keys=True).encode('utf-8')).hexdigest()
        return system

    @staticmethod
    @functools.lru_cache(None)
    def get_local_cache_path() -> Path:
        return Path(os.path.join(cache_dir(), 'cache', CacheBase.get_system()['hash']))

    @staticmethod
    @functools.lru_cache(None)
    def get_global_cache_path() -> Optional[Path]:
        return Path(os.path.join(config.global_cache_dir, CacheBase.get_system()['hash'])) if config.global_cache_dir is not None else None

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            return
        self.system = CacheBase.get_system()
        self.local_cache_path = CacheBase.get_local_cache_path()
        self.global_cache_path = CacheBase.get_global_cache_path()

    def get_local_cache(self) -> Dict[str, Any]:
        if not self.local_cache_path.is_file():
            return {}
        with open(self.local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache['cache']

    def update_local_cache(self, local_cache: Dict[str, Any]) -> None:
        if not os.path.exists(self.local_cache_path.parent):
            os.makedirs(self.local_cache_path.parent, exist_ok=True)
        write_atomic(str(self.local_cache_path), json.dumps({'system': self.system, 'cache': local_cache}, indent=4))