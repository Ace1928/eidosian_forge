import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
class NameClassMapContext(object):
    """Context manager to add/override in-place name->class maps."""

    def __init__(self, nameclassmap: NameClassMap, d: dict):
        self._nameclassmap = nameclassmap
        self._d = d
        self._keep: typing.List[typing.Tuple[str, bool, Optional[str]]] = []

    def __enter__(self):
        nameclassmap = self._nameclassmap
        for k, v in self._d.items():
            if k in nameclassmap:
                restore = True
                orig_v = nameclassmap[k]
            else:
                restore = False
                orig_v = None
            self._keep.append((k, restore, orig_v))
            nameclassmap[k] = v

    def __exit__(self, exc_type, exc_val, exc_tb):
        nameclassmap = self._nameclassmap
        for k, restore, orig_v in self._keep:
            if restore:
                nameclassmap[k] = orig_v
            else:
                del nameclassmap[k]
        return False