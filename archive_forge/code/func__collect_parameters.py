from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
def _collect_parameters(args):
    """Collect all type variables and parameter specifications in args
    in order of first appearance (lexicographic order).

    For example::

        >>> P = ParamSpec('P')
        >>> T = TypeVar('T')
        >>> _collect_parameters((T, Callable[P, T]))
        (~T, ~P)
    """
    parameters = []
    for t in args:
        if isinstance(t, type):
            pass
        elif isinstance(t, tuple):
            for x in t:
                for collected in _collect_parameters([x]):
                    if collected not in parameters:
                        parameters.append(collected)
        elif hasattr(t, '__typing_subst__'):
            if t not in parameters:
                parameters.append(t)
        else:
            for x in getattr(t, '__parameters__', ()):
                if x not in parameters:
                    parameters.append(x)
    return tuple(parameters)