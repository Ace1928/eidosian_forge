import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
class DebugContext:
    _counter = itertools.count()

    @staticmethod
    def wrap(fn):

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with DebugContext():
                return fn(*args, **kwargs)
        return wrap_compiler_debug(inner, compiler_name='inductor')

    @staticmethod
    def create_debug_dir(folder_name: str) -> Optional[str]:
        debug_dir = config.trace.debug_dir or get_debug_dir()
        for n in DebugContext._counter:
            dirname = os.path.join(debug_dir, 'torchinductor', f'{folder_name}.{n}')
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                return dirname
        return None

    def __init__(self):
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    def copy(self, new_path: str):
        if not self._path:
            return
        assert new_path.endswith('.debug'), new_path
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        try:
            shutil.copytree(self._path, new_path)
            self._path = new_path
        except OSError:
            log.warning('Failed to copy debug files from %s to %s', self._path, new_path)
            pass

    def fopen(self, filename: str):
        assert self._path
        return open(os.path.join(self._path, filename), 'w')

    def filename(self, suffix: str):
        assert self._path
        return os.path.join(self._path, suffix)

    def upload_tar(self):
        if config.trace.upload_tar is not None:
            import tarfile
            assert self._path
            tar_file = os.path.join(self._path, f'{os.path.basename(self._path)}.tar.gz')
            with tarfile.open(tar_file, 'w:gz') as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            config.trace.upload_tar(tar_file)

    def __enter__(self):
        if config.debug:
            log = logging.getLogger('torch._dynamo')
            prev_level = log.level
            log.setLevel(logging.DEBUG)

            def reset_log_level(level):
                log.setLevel(level)
            self._stack.callback(reset_log_level, prev_level)
        self._stack.enter_context(V.set_debug_handler(self))
        if not config.trace.enabled:
            return
        self._path = self.create_debug_dir(get_aot_graph_name())
        if config.trace.debug_log:
            self._setup_log_capture('debug.log', logging.DEBUG)
        if config.trace.info_log:
            self._setup_log_capture('info.log', logging.INFO)
        if config.trace.compile_profile:
            self._prof = cProfile.Profile()
            self._prof.enable()

    def _setup_log_capture(self, filename: str, level: int):
        log = logging.getLogger('torch._inductor')
        fd = self._stack.enter_context(self.fopen(filename))
        ch = logging.StreamHandler(fd)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('[%(filename)s:%(lineno)d %(levelname)s] %(message)s'))
        log.addHandler(ch)
        log.setLevel(min(log.level, level))
        self._stack.callback(log.removeHandler, ch)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prof:
            self._prof.disable()
            self._save_profile_data()
        if self._path:
            self.upload_tar()
            log.warning('%s debug trace: %s', get_graph_being_compiled(), self._path)
        self._stack.close()

    def _save_profile_data(self):
        assert self._prof
        self._prof.dump_stats(self.filename('compile.prof'))
        with self.fopen('compile.stats') as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            stats.strip_dirs()
            stats.sort_stats('cumtime')
            stats.print_stats(100)
            stats.sort_stats('tottime')
            stats.print_stats(100)

    def __getattr__(self, name):
        if config.trace.enabled and getattr(config.trace, name):
            try:
                return getattr(DebugFormatter(self), name)
            except Exception:
                log.warning('Ignoring exception in debug code', exc_info=True)
        else:

            def ignored(*args, **kwargs):
                pass
            return ignored