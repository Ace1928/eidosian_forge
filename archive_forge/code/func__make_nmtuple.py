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
def _make_nmtuple(name, types, module, defaults=()):
    fields = [n for n, t in types]
    annotations = {n: typing._type_check(t, f'field {n} annotation must be a type') for n, t in types}
    nm_tpl = collections.namedtuple(name, fields, defaults=defaults, module=module)
    nm_tpl.__annotations__ = nm_tpl.__new__.__annotations__ = annotations
    if sys.version_info < (3, 9):
        nm_tpl._field_types = annotations
    return nm_tpl