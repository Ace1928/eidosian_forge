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
@_tp_cache(typed=True)
def _class_getitem_inner(cls, *params):
    if len(params) < 2:
        raise TypeError('Annotated[...] should be used with at least two arguments (a type and an annotation).')
    if _is_unpacked_typevartuple(params[0]):
        raise TypeError('Annotated[...] should not be used with an unpacked TypeVarTuple')
    msg = 'Annotated[t, ...]: t must be a type.'
    origin = _type_check(params[0], msg, allow_special_forms=True)
    metadata = tuple(params[1:])
    return _AnnotatedAlias(origin, metadata)