import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
def _replace_arg(arg, tvars, args):
    """An internal helper function: replace arg if it is a type variable
    found in tvars with corresponding substitution from args or
    with corresponding substitution sub-tree if arg is a generic type.
    """
    if tvars is None:
        tvars = []
    if hasattr(arg, '_subs_tree') and isinstance(arg, (GenericMeta, _TypingBase)):
        return arg._subs_tree(tvars, args)
    if isinstance(arg, TypeVar):
        for i, tvar in enumerate(tvars):
            if arg == tvar:
                return args[i]
    return arg