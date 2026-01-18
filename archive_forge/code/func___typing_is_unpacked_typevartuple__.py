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
@property
def __typing_is_unpacked_typevartuple__(self):
    assert self.__origin__ is Unpack
    assert len(self.__args__) == 1
    return isinstance(self.__args__[0], TypeVarTuple)