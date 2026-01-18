import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _accept_device_cpu(func):

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        _check_device_cpu(kwargs.pop('device', None))
        return func(*args, **kwargs)
    return wrapped_func