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
class TritonFuture:
    kernel: ModuleType

    def __init__(self, kernel_name: str, source_code: str, future: Future[Any]) -> None:
        self.kernel_name = kernel_name
        self.source_code = source_code
        self.future = future

    def result(self) -> ModuleType:
        t0 = time()
        if hasattr(self, 'kernel'):
            return self.kernel
        self.future.result()
        kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code)
        latency = time() - t0
        if latency > 50:
            developer_warning(f'Detected long compilation time of {latency} seconds for kernel name {self.kernel_name}')
            developer_warning(self.source_code)
        del self.kernel_name, self.source_code, self.future
        return kernel