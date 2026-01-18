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
class _NamedTupleMeta(type):

    def __new__(cls, typename, bases, ns):
        assert _NamedTuple in bases
        for base in bases:
            if base is not _NamedTuple and base is not typing.Generic:
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
        if typing.Generic in bases:
            if hasattr(typing, '_generic_class_getitem'):
                nm_tpl.__class_getitem__ = classmethod(typing._generic_class_getitem)
            else:
                class_getitem = typing.Generic.__class_getitem__.__func__
                nm_tpl.__class_getitem__ = classmethod(class_getitem)
        for key, val in ns.items():
            if key in _prohibited_namedtuple_fields:
                raise AttributeError('Cannot overwrite NamedTuple attribute ' + key)
            elif key not in _special_namedtuple_fields:
                if key not in nm_tpl._fields:
                    setattr(nm_tpl, key, ns[key])
                try:
                    set_name = type(val).__set_name__
                except AttributeError:
                    pass
                else:
                    try:
                        set_name(val, nm_tpl, key)
                    except BaseException as e:
                        msg = f'Error calling __set_name__ on {type(val).__name__!r} instance {key!r} in {typename!r}'
                        if sys.version_info >= (3, 12):
                            e.add_note(msg)
                            raise
                        else:
                            raise RuntimeError(msg) from e
        if typing.Generic in bases:
            nm_tpl.__init_subclass__()
        return nm_tpl