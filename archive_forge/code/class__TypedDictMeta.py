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
class _TypedDictMeta(type):

    def __new__(cls, name, bases, ns, *, total=True, closed=False):
        """Create new typed dict class object.

            This method is called when TypedDict is subclassed,
            or when TypedDict is instantiated. This way
            TypedDict supports all three syntax forms described in its docstring.
            Subclasses and instances of TypedDict return actual dictionaries.
            """
        for base in bases:
            if type(base) is not _TypedDictMeta and base is not typing.Generic:
                raise TypeError('cannot inherit from both a TypedDict type and a non-TypedDict base class')
        if any((issubclass(b, typing.Generic) for b in bases)):
            generic_base = (typing.Generic,)
        else:
            generic_base = ()
        tp_dict = type.__new__(_TypedDictMeta, 'Protocol', (*generic_base, dict), ns)
        tp_dict.__name__ = name
        if tp_dict.__qualname__ == 'Protocol':
            tp_dict.__qualname__ = name
        if not hasattr(tp_dict, '__orig_bases__'):
            tp_dict.__orig_bases__ = bases
        annotations = {}
        own_annotations = ns.get('__annotations__', {})
        msg = "TypedDict('Name', {f0: t0, f1: t1, ...}); each t must be a type"
        if _TAKES_MODULE:
            own_annotations = {n: typing._type_check(tp, msg, module=tp_dict.__module__) for n, tp in own_annotations.items()}
        else:
            own_annotations = {n: typing._type_check(tp, msg) for n, tp in own_annotations.items()}
        required_keys = set()
        optional_keys = set()
        readonly_keys = set()
        mutable_keys = set()
        extra_items_type = None
        for base in bases:
            base_dict = base.__dict__
            annotations.update(base_dict.get('__annotations__', {}))
            required_keys.update(base_dict.get('__required_keys__', ()))
            optional_keys.update(base_dict.get('__optional_keys__', ()))
            readonly_keys.update(base_dict.get('__readonly_keys__', ()))
            mutable_keys.update(base_dict.get('__mutable_keys__', ()))
            base_extra_items_type = base_dict.get('__extra_items__', None)
            if base_extra_items_type is not None:
                extra_items_type = base_extra_items_type
        if closed and extra_items_type is None:
            extra_items_type = Never
        if closed and '__extra_items__' in own_annotations:
            annotation_type = own_annotations.pop('__extra_items__')
            qualifiers = set(_get_typeddict_qualifiers(annotation_type))
            if Required in qualifiers:
                raise TypeError('Special key __extra_items__ does not support Required')
            if NotRequired in qualifiers:
                raise TypeError('Special key __extra_items__ does not support NotRequired')
            extra_items_type = annotation_type
        annotations.update(own_annotations)
        for annotation_key, annotation_type in own_annotations.items():
            qualifiers = set(_get_typeddict_qualifiers(annotation_type))
            if Required in qualifiers:
                required_keys.add(annotation_key)
            elif NotRequired in qualifiers:
                optional_keys.add(annotation_key)
            elif total:
                required_keys.add(annotation_key)
            else:
                optional_keys.add(annotation_key)
            if ReadOnly in qualifiers:
                mutable_keys.discard(annotation_key)
                readonly_keys.add(annotation_key)
            else:
                mutable_keys.add(annotation_key)
                readonly_keys.discard(annotation_key)
        tp_dict.__annotations__ = annotations
        tp_dict.__required_keys__ = frozenset(required_keys)
        tp_dict.__optional_keys__ = frozenset(optional_keys)
        tp_dict.__readonly_keys__ = frozenset(readonly_keys)
        tp_dict.__mutable_keys__ = frozenset(mutable_keys)
        if not hasattr(tp_dict, '__total__'):
            tp_dict.__total__ = total
        tp_dict.__closed__ = closed
        tp_dict.__extra_items__ = extra_items_type
        return tp_dict
    __call__ = dict

    def __subclasscheck__(cls, other):
        raise TypeError('TypedDict does not support instance and class checks')
    __instancecheck__ = __subclasscheck__