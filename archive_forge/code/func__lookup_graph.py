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
def _lookup_graph(key: str, example_inputs: List[torch.Tensor]) -> Optional[CompiledFxGraph]:
    """
        Lookup a compiled graph in the cache by key. On a hit, return the
        deserialized CompiledFxGraph object. On a miss, return None.
        """
    subdir = FxGraphCache._get_tmp_dir_for_key(key)
    if not os.path.exists(subdir):
        return None
    for path in sorted(os.listdir(subdir)):
        with open(os.path.join(subdir, path), 'rb') as f:
            graph: CompiledFxGraph = pickle.load(f)
        guards_expr = graph.guards_expr
        if not guards_expr:
            return graph
        shape_env = FxGraphCache._get_shape_env()
        symints = FxGraphCache._filter_symints(example_inputs)
        assert all((has_hint(s) for s in symints))
        hints = [hint_int(s) for s in symints]
        hit = bool(shape_env.evaluate_guards_expression(guards_expr, hints))
        log.debug('fx graph cache key %s evaluating guards for %s with values %s => %s', key, guards_expr, hints, hit)
        if hit:
            check = bool(shape_env.evaluate_guards_expression(guards_expr, symints))
            assert check is True
            log.debug('fx graph cache key %s post-load guards: %s', key, shape_env.guards)
            return graph
    return None