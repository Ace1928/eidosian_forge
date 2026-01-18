import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _dispatch_opts(self, *args, **kwargs):
    if self._mode is None:
        return self._base_opts(*args, **kwargs)
    elif self._mode == 'holomap':
        return self._holomap_opts(*args, **kwargs)
    elif self._mode == 'dynamicmap':
        return self._dynamicmap_opts(*args, **kwargs)