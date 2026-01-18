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
class _ExtensionsSpecialForm(typing._SpecialForm, _root=True):

    def __repr__(self):
        return 'typing_extensions.' + self._name