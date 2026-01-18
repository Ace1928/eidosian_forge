import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
class TestParallelBackendBase(TestCase):
    """
    Base class for testing the parallel backends
    """
    all_impls = [jit_runner(nopython=True), jit_runner(nopython=True, cache=True), jit_runner(nopython=True, nogil=True), linalg_runner(nopython=True), linalg_runner(nopython=True, nogil=True), vectorize_runner(nopython=True), vectorize_runner(nopython=True, target='parallel'), vectorize_runner(nopython=True, target='parallel', cache=True), guvectorize_runner(nopython=True), guvectorize_runner(nopython=True, target='parallel'), guvectorize_runner(nopython=True, target='parallel', cache=True)]
    if not _parfors_unsupported:
        parfor_impls = [jit_runner(nopython=True, parallel=True), jit_runner(nopython=True, parallel=True, cache=True), linalg_runner(nopython=True, parallel=True), linalg_runner(nopython=True, parallel=True, cache=True)]
        all_impls.extend(parfor_impls)
    if config.NUMBA_NUM_THREADS < 2:
        masks = []
    else:
        masks = [1, 2]
    mask_impls = []
    for impl in all_impls:
        for mask in masks:
            mask_impls.append(mask_runner(impl, mask))
    parallelism = ['threading', 'random']
    parallelism.append('multiprocessing_spawn')
    if _HAVE_OS_FORK:
        parallelism.append('multiprocessing_fork')
        parallelism.append('multiprocessing_forkserver')
    runners = {'concurrent_jit': [jit_runner(nopython=True, parallel=not _parfors_unsupported)], 'concurrent_vectorize': [vectorize_runner(nopython=True, target='parallel')], 'concurrent_guvectorize': [guvectorize_runner(nopython=True, target='parallel')], 'concurrent_mix_use': all_impls, 'concurrent_mix_use_masks': mask_impls}
    safe_backends = {'omp', 'tbb'}

    def run_compile(self, fnlist, parallelism='threading'):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            if parallelism == 'threading':
                thread_impl(fnlist)
            elif parallelism == 'multiprocessing_fork':
                fork_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_forkserver':
                forkserver_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_spawn':
                spawn_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_default':
                default_proc_impl(fnlist)
            elif parallelism == 'random':
                ps = [thread_impl, spawn_proc_impl]
                if _HAVE_OS_FORK:
                    ps.append(fork_proc_impl)
                    ps.append(forkserver_proc_impl)
                random.shuffle(ps)
                for impl in ps:
                    impl(fnlist)
            else:
                raise ValueError('Unknown parallelism supplied %s' % parallelism)