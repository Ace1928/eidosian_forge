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
class PersistentCache(CacheBase):

    @functools.lru_cache(None)
    def get_global_cache(self):
        if self.global_cache_path is None or not self.global_cache_path.is_file():
            return {}
        with open(self.global_cache_path) as global_cache_fp:
            global_cache = json.load(global_cache_fp)
        return global_cache['cache']

    def lookup(self, choices: List[ChoiceCaller], name: str, inputs: str, benchmark: Callable[[Any], Dict[ChoiceCaller, float]]) -> Dict[ChoiceCaller, float]:
        """
        Check to see if we have benchmarked the given choice callers. For each
        choice caller:

            1. Check global_cache[name][inputs][choice], return benchmark if cached.
            2. Check local_cache[name][inputs][choice], return benchmark if cached.
            3.
                a. `max_autotune_gemm=True`: benchmark the choice, update
                    local_cache[name][inputs][choice], and return the benchmark.
                b. `max_autotune_gemm=False`: don't benchmark the choice, return nothing.
        """
        log_stats = partial(log_global_cache_stats, self.system, name, inputs)
        log_vals = partial(log_global_cache_vals, self.system, name, inputs)
        log_errors = partial(log_global_cache_errors, self.system, name, inputs)
        timings = {}

        def check_cache(cache, callback=None) -> bool:
            """Check if `cache` contains data for all the choices"""
            hit = True
            for choice in choices:
                choice_hash = choice.hash_key()
                if choice_hash in cache.get(name, {}).get(inputs, {}):
                    timings[choice] = cache[name][inputs][choice_hash]
                else:
                    hit = False
                    break
            if callback:
                callback(cached=hit)
            return hit
        if config.max_autotune or config.max_autotune_gemm:
            local_cache = self.get_local_cache()
            if not check_cache(local_cache) and (not (use_global_cache() and check_cache(self.get_global_cache(), callback=log_stats))):
                try:
                    timings = benchmark(choices)
                    assert all((choice in timings for choice in choices))
                    local_cache.setdefault(name, {})
                    local_cache[name].setdefault(inputs, {})
                    for choice, timing in timings.items():
                        local_cache[name][inputs][choice.hash_key()] = timing
                except RuntimeError as e:
                    log_errors(e)
                    raise e
                self.update_local_cache(local_cache)
                timings_to_log = {choice.hash_key(): timings[choice] for choice in choices}
                log_vals(timings_to_log)
        elif use_global_cache():
            check_cache(self.get_global_cache(), callback=log_stats)
        return timings