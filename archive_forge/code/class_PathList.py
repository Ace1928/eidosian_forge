import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
class PathList(Path):
    """
    Domain validator for a list of
    :py:term:`path-like objects <path-like object>`.

    This admits a path-like object or iterable of such.
    If a path-like object is passed, then
    a singleton list containing the object normalized through
    :py:class:`Path` is returned.
    An iterable of path-like objects is cast to a list, each
    entry of which is normalized through :py:class:`Path`.

    Parameters
    ----------
    basePath: Union[None, str, ConfigValue]
        The base path that will be prepended to any non-absolute path
        values provided.  If None, defaults to :py:attr:`Path.BasePath`.

    expandPath: bool
        If True, then the value will be expanded and normalized.  If
        False, the string representation of the value will be returned
        unchanged.  If None, expandPath will defer to the (negated)
        value of :py:attr:`Path.SuppressPathExpansion`

    """

    def __call__(self, data):
        is_path_like = isinstance(data, (str, bytes)) or hasattr(data, '__fspath__')
        if hasattr(data, '__iter__') and (not is_path_like):
            return [super(PathList, self).__call__(i) for i in data]
        else:
            return [super(PathList, self).__call__(data)]