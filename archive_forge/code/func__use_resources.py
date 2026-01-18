import os
import numpy as np
import threading
from time import time
from .. import config, logging
def _use_resources(n_procs, mem_gb):
    """
    Function to execute multiple use_gb_ram functions in parallel
    """
    import os
    import sys
    import psutil
    from multiprocessing import Pool
    from nipype import logging
    from nipype.utils.profiler import _use_cpu
    iflogger = logging.getLogger('nipype.interface')
    BSIZE = sys.getsizeof('  ') - sys.getsizeof(' ')
    BOFFSET = sys.getsizeof('')
    _GB = 1024.0 ** 3

    def _use_gb_ram(mem_gb):
        """A test function to consume mem_gb GB of RAM"""
        num_bytes = int(mem_gb * _GB)
        gb_str = ' ' * ((num_bytes - BOFFSET) // BSIZE)
        assert sys.getsizeof(gb_str) == num_bytes
        return gb_str
    p = psutil.Process(os.getpid())
    mem_offset = p.memory_info().rss / _GB
    big_str = _use_gb_ram(mem_gb - mem_offset)
    _use_cpu(5)
    mem_total = p.memory_info().rss / _GB
    del big_str
    iflogger.info('[%d] Memory offset %0.2fGB, total %0.2fGB', os.getpid(), mem_offset, mem_total)
    if n_procs > 1:
        pool = Pool(n_procs)
        pool.map(_use_cpu, range(n_procs))
    return True