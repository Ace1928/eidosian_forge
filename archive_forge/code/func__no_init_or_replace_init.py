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
def _no_init_or_replace_init(self, *args, **kwargs):
    cls = type(self)
    if cls._is_protocol:
        raise TypeError('Protocols cannot be instantiated')
    if cls.__init__ is not _no_init_or_replace_init:
        return
    for base in cls.__mro__:
        init = base.__dict__.get('__init__', _no_init_or_replace_init)
        if init is not _no_init_or_replace_init:
            cls.__init__ = init
            break
    else:
        cls.__init__ = object.__init__
    cls.__init__(self, *args, **kwargs)