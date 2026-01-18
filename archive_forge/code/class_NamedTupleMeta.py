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
class NamedTupleMeta(type):

    def __new__(cls, typename, bases, ns):
        assert _NamedTuple in bases
        for base in bases:
            if base is not _NamedTuple and base is not Generic:
                raise TypeError('can only inherit from a NamedTuple type and Generic')
        bases = tuple((tuple if base is _NamedTuple else base for base in bases))
        types = ns.get('__annotations__', {})
        default_names = []
        for field_name in types:
            if field_name in ns:
                default_names.append(field_name)
            elif default_names:
                raise TypeError(f'Non-default namedtuple field {field_name} cannot follow default field{('s' if len(default_names) > 1 else '')} {', '.join(default_names)}')
        nm_tpl = _make_nmtuple(typename, types.items(), defaults=[ns[n] for n in default_names], module=ns['__module__'])
        nm_tpl.__bases__ = bases
        if Generic in bases:
            class_getitem = Generic.__class_getitem__.__func__
            nm_tpl.__class_getitem__ = classmethod(class_getitem)
        for key in ns:
            if key in _prohibited:
                raise AttributeError('Cannot overwrite NamedTuple attribute ' + key)
            elif key not in _special and key not in nm_tpl._fields:
                setattr(nm_tpl, key, ns[key])
        if Generic in bases:
            nm_tpl.__init_subclass__()
        return nm_tpl