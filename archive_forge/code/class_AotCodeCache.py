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
class AotCodeCache:
    cache: Dict[str, str] = dict()
    clear = staticmethod(cache.clear)

    @classmethod
    def compile(cls, graph: GraphLowering, source_code: str, serialized_extern_kernel_nodes: Optional[str], cuda: bool) -> str:
        picked_vec_isa = pick_vec_isa()
        cpp_command = repr(cpp_compile_command('i', 'o', vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode))
        fbcode_aot_cpu_re = False
        use_absolute_path = False
        if config.is_fbcode():
            ld_command = build_paths.ld()
            if not cuda and graph.aot_mode:
                objcopy_command = build_paths.objcopy_fallback()
                fbcode_aot_cpu_re = True
                use_absolute_path = True
            else:
                objcopy_command = build_paths.objcopy()
        else:
            ld_command = 'ld'
            objcopy_command = 'objcopy'
        specified_output_path, specified_so_name = split_aot_inductor_output_path(config.aot_inductor.output_path)
        key, input_path = write(source_code, 'cpp', extra=cpp_command, specified_dir=specified_output_path)
        if key not in cls.cache or (specified_output_path and os.path.dirname(cls.cache[key]) != specified_output_path or (specified_so_name and os.path.basename(cls.cache[key]) != specified_so_name)):
            from filelock import FileLock
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + '.lock'), timeout=LOCK_TIMEOUT)
            with lock:
                if config.is_fbcode() and serialized_extern_kernel_nodes:
                    output_json = os.path.splitext(input_path)[0] + '.json'
                    with open(output_json, 'w') as f:
                        f.write(serialized_extern_kernel_nodes)
                output_so = config.aot_inductor.output_path if specified_so_name else os.path.splitext(input_path)[0] + '.so'
                if not os.path.exists(output_so):
                    output_o = os.path.splitext(input_path)[0] + '.o'
                    cmd = cpp_compile_command(input=input_path, output=output_o, vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode, compile_only=True, use_absolute_path=use_absolute_path)
                    log.debug('aot compilation command: %s', cmd)
                    if fbcode_aot_cpu_re:
                        compile_file(input_path, output_o, cmd.split())
                        os.chmod(output_o, 420)
                    else:
                        run_command_and_check(cmd)

                    def _to_bytes(t: torch.Tensor) -> bytes:
                        import ctypes
                        if t.numel() == 0:
                            return b''
                        t_cpu = t.untyped_storage().cpu()
                        raw_array = ctypes.cast(t_cpu.data_ptr(), ctypes.POINTER(ctypes.c_ubyte * t_cpu.nbytes()))
                        return bytes(raw_array.contents)
                    aot_constants = b''.join((_to_bytes(tensor) for tensor in graph.constants.values()))
                    consts_key, consts_path = write(aot_constants, 'bin', specified_dir=specified_output_path)
                    consts_o = os.path.splitext(consts_path)[0] + '.o'
                    if fbcode_aot_cpu_re:
                        cmd = f'{ld_command} -r -b binary -o {os.path.basename(consts_o)} {os.path.basename(consts_path)}'
                        compile_file(consts_path, consts_o, cmd.split())
                        os.chmod(consts_o, 420)
                    else:
                        cmd = f'{ld_command} -r -b binary -o {consts_o} {consts_path}'
                        run_command_and_check(cmd)
                    log.debug('aot constant binary command: %s', cmd)
                    cmd = f'{objcopy_command} --rename-section .data=.lrodata,alloc,load,readonly,data,contents {consts_o} {consts_o}'
                    log.debug('aot constant obj command: %s', cmd)
                    run_command_and_check(cmd)
                    cmd = f'rm {consts_path}'
                    log.debug('aot constant bin removal command: %s', cmd)
                    run_command_and_check(cmd)
                    if fbcode_aot_cpu_re:
                        body = re.sub('[\\W]', '_', os.path.basename(consts_path))
                    else:
                        body = re.sub('[\\W]', '_', consts_path)
                    symbol_list = []
                    symbol_list.append(f'{objcopy_command} --redefine-sym _binary_{body}_start=_binary_constants_bin_start {consts_o}')
                    symbol_list.append(f'{objcopy_command} --redefine-sym _binary_{body}_size=_binary_constants_bin_size {consts_o}')
                    symbol_list.append(f'{objcopy_command} --redefine-sym _binary_{body}_end=_binary_constants_bin_end {consts_o}')
                    log.debug('aot constant binary redefine symbol: %s', ' '.join(symbol_list))
                    for cmd in symbol_list:
                        run_command_and_check(cmd)
                    cmd = cpp_compile_command(input=[output_o, consts_o], output=output_so, vec_isa=picked_vec_isa, cuda=cuda, aot_mode=graph.aot_mode, use_absolute_path=use_absolute_path)
                    log.debug('aot linkage command: %s', cmd)
                    if fbcode_aot_cpu_re:
                        compile_file([output_o, consts_o], output_so, cmd.split())
                        os.chmod(output_so, 493)
                    else:
                        run_command_and_check(cmd)
                else:
                    log.debug('aot_inductor dynamic library already exist: %s', output_so)
                cls.cache[key] = output_so
        return cls.cache[key]