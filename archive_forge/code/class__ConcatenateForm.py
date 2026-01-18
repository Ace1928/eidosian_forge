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
class _ConcatenateForm(_ExtensionsSpecialForm, _root=True):

    def __getitem__(self, parameters):
        return _concatenate_getitem(self, parameters)