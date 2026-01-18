import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
class _UnpackSpecialForm(_ExtensionsSpecialForm, _root=True):

    def __init__(self, getitem):
        super().__init__(getitem)
        self.__doc__ = _UNPACK_DOC