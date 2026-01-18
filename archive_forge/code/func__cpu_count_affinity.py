import os
import sys
import math
import subprocess
import traceback
import warnings
import multiprocessing as mp
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from .process import LokyProcess, LokyInitMainProcess
def _cpu_count_affinity(os_cpu_count):
    if hasattr(os, 'sched_getaffinity'):
        try:
            return len(os.sched_getaffinity(0))
        except NotImplementedError:
            pass
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p, 'cpu_affinity'):
            return len(p.cpu_affinity())
    except ImportError:
        if sys.platform == 'linux' and os.environ.get('LOKY_MAX_CPU_COUNT') is None:
            warnings.warn('Failed to inspect CPU affinity constraints on this system. Please install psutil or explictly set LOKY_MAX_CPU_COUNT.')
    return os_cpu_count