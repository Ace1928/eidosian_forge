import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class _Optional(_FinalTypingBase, _root=True):
    """Optional type.

    Optional[X] is equivalent to Union[X, None].
    """
    __slots__ = ()

    @_tp_cache
    def __getitem__(self, arg):
        arg = _type_check(arg, 'Optional[t] requires a single type.')
        return Union[arg, type(None)]