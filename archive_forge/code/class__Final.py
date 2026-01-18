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
class _Final:
    """Mixin to prohibit subclassing."""
    __slots__ = ('__weakref__',)

    def __init_subclass__(cls, /, *args, **kwds):
        if '_root' not in kwds:
            raise TypeError('Cannot subclass special typing classes')