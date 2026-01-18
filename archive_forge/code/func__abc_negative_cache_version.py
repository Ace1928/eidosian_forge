import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
@_abc_negative_cache_version.setter
def _abc_negative_cache_version(self, value):
    if self.__origin__ is None:
        if isinstance(self.__extra__, abc.ABCMeta):
            self.__extra__._abc_negative_cache_version = value
        else:
            self._abc_generic_negative_cache_version = value