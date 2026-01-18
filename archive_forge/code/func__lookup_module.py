import re
import os
import sys
import warnings
from dill import _dill, Pickler, Unpickler
from ._dill import (
from typing import Optional, Union
import pathlib
import tempfile
def _lookup_module(modmap, name, obj, main_module):
    """lookup name or id of obj if module is imported"""
    for modobj, modname in modmap.by_name[name]:
        if modobj is obj and sys.modules[modname] is not main_module:
            return (modname, name)
    __module__ = getattr(obj, '__module__', None)
    if isinstance(obj, IMPORTED_AS_TYPES) or (__module__ is not None and any((regex.fullmatch(__module__) for regex in IMPORTED_AS_MODULES))):
        for modobj, objname, modname in modmap.by_id[id(obj)]:
            if sys.modules[modname] is not main_module:
                return (modname, objname)
    return (None, None)