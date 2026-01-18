from numbers import Number
import operator
import os
import threading
import contextlib
import numpy as np
from .pypocketfft import good_size
def _workers(workers):
    if workers is None:
        return getattr(_config, 'default_workers', 1)
    if workers < 0:
        if workers >= -_cpu_count:
            workers += 1 + _cpu_count
        else:
            raise ValueError(f'workers value out of range; got {workers}, must not be less than {-_cpu_count}')
    elif workers == 0:
        raise ValueError('workers must not be zero')
    return workers