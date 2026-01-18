from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _readline_side_effect():
    if handle.readline.return_value is not None:
        while True:
            yield handle.readline.return_value
    for line in _state[0]:
        yield line