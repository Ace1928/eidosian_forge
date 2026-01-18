import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
def _get_metric_name(fn):
    qualname = fn.__qualname__
    split = qualname.split('.')
    if len(split) == 1:
        module = fn.__module__
        if module:
            return module.split('.')[-1] + '.' + split[0]
        else:
            return split[0]
    else:
        return qualname