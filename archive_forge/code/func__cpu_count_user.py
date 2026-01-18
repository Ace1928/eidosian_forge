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
def _cpu_count_user(os_cpu_count):
    """Number of user defined available CPUs"""
    cpu_count_affinity = _cpu_count_affinity(os_cpu_count)
    cpu_count_cgroup = _cpu_count_cgroup(os_cpu_count)
    cpu_count_loky = int(os.environ.get('LOKY_MAX_CPU_COUNT', os_cpu_count))
    return min(cpu_count_affinity, cpu_count_cgroup, cpu_count_loky)