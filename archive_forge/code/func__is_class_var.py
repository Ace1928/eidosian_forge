import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _is_class_var(annot):
    """
    Check whether *annot* is a typing.ClassVar.

    The string comparison hack is used to avoid evaluating all string
    annotations which would put attrs-based classes at a performance
    disadvantage compared to plain old classes.
    """
    annot = str(annot)
    if annot.startswith(("'", '"')) and annot.endswith(("'", '"')):
        annot = annot[1:-1]
    return annot.startswith(_classvar_prefixes)