from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types
from funcsigs.version import __version__
class _ParameterKind(int):

    def __new__(self, *args, **kwargs):
        obj = int.__new__(self, *args)
        obj._name = kwargs['name']
        return obj

    def __str__(self):
        return self._name

    def __repr__(self):
        return '<_ParameterKind: {0!r}>'.format(self._name)