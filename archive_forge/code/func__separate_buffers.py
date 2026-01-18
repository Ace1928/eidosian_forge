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
def _separate_buffers(substate, path, buffer_paths, buffers):
    """For internal, see _remove_buffers"""
    if isinstance(substate, (list, tuple)):
        is_cloned = False
        for i, v in enumerate(substate):
            if isinstance(v, _binary_types):
                if not is_cloned:
                    substate = list(substate)
                    is_cloned = True
                substate[i] = None
                buffers.append(v)
                buffer_paths.append(path + [i])
            elif isinstance(v, (dict, list, tuple)):
                vnew = _separate_buffers(v, path + [i], buffer_paths, buffers)
                if v is not vnew:
                    if not is_cloned:
                        substate = list(substate)
                        is_cloned = True
                    substate[i] = vnew
    elif isinstance(substate, dict):
        is_cloned = False
        for k, v in substate.items():
            if isinstance(v, _binary_types):
                if not is_cloned:
                    substate = dict(substate)
                    is_cloned = True
                del substate[k]
                buffers.append(v)
                buffer_paths.append(path + [k])
            elif isinstance(v, (dict, list, tuple)):
                vnew = _separate_buffers(v, path + [k], buffer_paths, buffers)
                if v is not vnew:
                    if not is_cloned:
                        substate = dict(substate)
                        is_cloned = True
                    substate[k] = vnew
    else:
        raise ValueError('expected state to be a list or dict, not %r' % substate)
    return substate