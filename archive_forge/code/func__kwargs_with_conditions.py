import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def _kwargs_with_conditions(kwargs, method):
    if method and 'conditions' not in kwargs:
        newkwargs = kwargs.copy()
        newkwargs['conditions'] = {'method': method}
        return newkwargs
    else:
        return kwargs