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
class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """
    EXCLUDED_KWARGS = ['graph_id']

    def __init__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], fx_kwargs: Dict[str, Any]):
        self.gm = gm
        self.example_inputs = example_inputs
        self.fx_kwargs = {}
        for k in sorted(fx_kwargs):
            if k not in self.EXCLUDED_KWARGS:
                if type(fx_kwargs[k]) is set:
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(fx_kwargs[k]))
                else:
                    self.fx_kwargs[k] = fx_kwargs[k]
        self.torch_version = torch.__version__
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config()
        self.inductor_code_hash = get_inductor_code_hash()

    def debug_str(self) -> str:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata(obj))
            elif isinstance(obj, bytes):
                return '<bytes>'
            else:
                return str(obj)
        lines = []
        for attr, obj in vars(self).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = FxGraphCachePickler.get_hash(obj[ii])
                    lines.append(f'[{h}] {attr}[{ii}]: {get_str(obj[ii])}')
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = FxGraphCachePickler.get_hash(v)
                    lines.append(f'[{h}] {attr}[{k}]: {get_str(v)}')
            else:
                h = FxGraphCachePickler.get_hash(obj)
                lines.append(f'[{h}] {attr}: {get_str(obj)}')
        return '\n'.join(lines)