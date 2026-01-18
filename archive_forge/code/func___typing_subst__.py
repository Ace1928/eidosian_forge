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
def __typing_subst__(self, arg):
    if isinstance(arg, (list, tuple)):
        arg = tuple((_type_check(a, 'Expected a type.') for a in arg))
    elif not _is_param_expr(arg):
        raise TypeError(f'Expected a list of types, an ellipsis, ParamSpec, or Concatenate. Got {arg}')
    return arg