import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def _get_objects_path(self):

    def get():
        obj = self._obj
        yield obj
        try:
            obj = obj.__objclass__
        except AttributeError:
            pass
        else:
            yield obj
        try:
            imp_plz = obj.__module__
        except AttributeError:
            if not inspect.ismodule(obj):
                yield builtins
        else:
            if imp_plz is None:
                yield builtins
            else:
                try:
                    yield sys.modules[imp_plz]
                except KeyError:
                    yield builtins
    return list(reversed(list(get())))