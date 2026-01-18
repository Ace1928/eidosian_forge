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
@typing._tp_cache
def _concatenate_getitem(self, parameters):
    if parameters == ():
        raise TypeError('Cannot take a Concatenate of no types.')
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    if not isinstance(parameters[-1], ParamSpec):
        raise TypeError('The last parameter to Concatenate should be a ParamSpec variable.')
    msg = 'Concatenate[arg, ...]: each arg must be a type.'
    parameters = tuple((typing._type_check(p, msg) for p in parameters))
    return _ConcatenateGenericAlias(self, parameters)