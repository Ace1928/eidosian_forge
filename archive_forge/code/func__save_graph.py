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
@staticmethod
def _save_graph(key: str, compiled_graph: CompiledFxGraph, example_inputs: List[torch.Tensor]):
    """
        Store a serialized CompiledFxGraph on disk.
        """
    disk_compiled_graph = copy(compiled_graph)
    disk_compiled_graph.compiled_artifact = None
    shape_env = FxGraphCache._get_shape_env()
    symints = FxGraphCache._filter_symints(example_inputs)
    disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(symints)
    content = pickle.dumps(disk_compiled_graph)
    subdir = FxGraphCache._get_tmp_dir_for_key(key)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, sha256_hash(content))
    write_atomic(path, content)