from __future__ import absolute_import
from __future__ import print_function
from collections import namedtuple
import copy
import hashlib
import os
import six
def _KeyValueToDict(pair):
    """Converts an iterable object of key=value pairs to dictionary."""
    d = dict()
    for kv in pair:
        k, v = kv.split('=', 1)
        d[k] = v
    return d