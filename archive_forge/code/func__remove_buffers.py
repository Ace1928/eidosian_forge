import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
def _remove_buffers(state):
    """Return (state_without_buffers, buffer_paths, buffers) for binary message parts

    A binary message part is a memoryview, bytearray, or python 3 bytes object.

    As an example:
    >>> state = {'plain': [0, 'text'], 'x': {'ar': memoryview(ar1)}, 'y': {'shape': (10,10), 'data': memoryview(ar2)}}
    >>> _remove_buffers(state)
    ({'plain': [0, 'text']}, {'x': {}, 'y': {'shape': (10, 10)}}, [['x', 'ar'], ['y', 'data']],
     [<memory at 0x107ffec48>, <memory at 0x107ffed08>])
    """
    buffer_paths, buffers = ([], [])
    state = _separate_buffers(state, [], buffer_paths, buffers)
    return (state, buffer_paths, buffers)