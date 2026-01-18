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
def _count_physical_cores():
    """Return a tuple (number of physical cores, exception)

    If the number of physical cores is found, exception is set to None.
    If it has not been found, return ("not found", exception).

    The number of physical cores is cached to avoid repeating subprocess calls.
    """
    exception = None
    global physical_cores_cache
    if physical_cores_cache is not None:
        return (physical_cores_cache, exception)
    try:
        if sys.platform == 'linux':
            cpu_info = subprocess.run('lscpu --parse=core'.split(), capture_output=True, text=True)
            cpu_info = cpu_info.stdout.splitlines()
            cpu_info = {line for line in cpu_info if not line.startswith('#')}
            cpu_count_physical = len(cpu_info)
        elif sys.platform == 'win32':
            cpu_info = subprocess.run('wmic CPU Get NumberOfCores /Format:csv'.split(), capture_output=True, text=True)
            cpu_info = cpu_info.stdout.splitlines()
            cpu_info = [l.split(',')[1] for l in cpu_info if l and l != 'Node,NumberOfCores']
            cpu_count_physical = sum(map(int, cpu_info))
        elif sys.platform == 'darwin':
            cpu_info = subprocess.run('sysctl -n hw.physicalcpu'.split(), capture_output=True, text=True)
            cpu_info = cpu_info.stdout
            cpu_count_physical = int(cpu_info)
        else:
            raise NotImplementedError(f'unsupported platform: {sys.platform}')
        if cpu_count_physical < 1:
            raise ValueError(f'found {cpu_count_physical} physical cores < 1')
    except Exception as e:
        exception = e
        cpu_count_physical = 'not found'
    physical_cores_cache = cpu_count_physical
    return (cpu_count_physical, exception)