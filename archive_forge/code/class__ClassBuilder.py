import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
class _ClassBuilder:
    """
    Iteratively build *one* class.
    """
    __slots__ = ('_attr_names', '_attrs', '_base_attr_map', '_base_names', '_cache_hash', '_cls', '_cls_dict', '_delete_attribs', '_frozen', '_has_pre_init', '_pre_init_has_args', '_has_post_init', '_is_exc', '_on_setattr', '_slots', '_weakref_slot', '_wrote_own_setattr', '_has_custom_setattr')

    def __init__(self, cls, these, slots, frozen, weakref_slot, getstate_setstate, auto_attribs, kw_only, cache_hash, is_exc, collect_by_mro, on_setattr, has_custom_setattr, field_transformer):
        attrs, base_attrs, base_map = _transform_attrs(cls, these, auto_attribs, kw_only, collect_by_mro, field_transformer)
        self._cls = cls
        self._cls_dict = dict(cls.__dict__) if slots else {}
        self._attrs = attrs
        self._base_names = {a.name for a in base_attrs}
        self._base_attr_map = base_map
        self._attr_names = tuple((a.name for a in attrs))
        self._slots = slots
        self._frozen = frozen
        self._weakref_slot = weakref_slot
        self._cache_hash = cache_hash
        self._has_pre_init = bool(getattr(cls, '__attrs_pre_init__', False))
        self._pre_init_has_args = False
        if self._has_pre_init:
            pre_init_func = cls.__attrs_pre_init__
            pre_init_signature = inspect.signature(pre_init_func)
            self._pre_init_has_args = len(pre_init_signature.parameters) > 1
        self._has_post_init = bool(getattr(cls, '__attrs_post_init__', False))
        self._delete_attribs = not bool(these)
        self._is_exc = is_exc
        self._on_setattr = on_setattr
        self._has_custom_setattr = has_custom_setattr
        self._wrote_own_setattr = False
        self._cls_dict['__attrs_attrs__'] = self._attrs
        if frozen:
            self._cls_dict['__setattr__'] = _frozen_setattrs
            self._cls_dict['__delattr__'] = _frozen_delattrs
            self._wrote_own_setattr = True
        elif on_setattr in (_ng_default_on_setattr, setters.validate, setters.convert):
            has_validator = has_converter = False
            for a in attrs:
                if a.validator is not None:
                    has_validator = True
                if a.converter is not None:
                    has_converter = True
                if has_validator and has_converter:
                    break
            if on_setattr == _ng_default_on_setattr and (not (has_validator or has_converter)) or (on_setattr == setters.validate and (not has_validator)) or (on_setattr == setters.convert and (not has_converter)):
                self._on_setattr = None
        if getstate_setstate:
            self._cls_dict['__getstate__'], self._cls_dict['__setstate__'] = self._make_getstate_setstate()

    def __repr__(self):
        return f'<_ClassBuilder(cls={self._cls.__name__})>'
    if PY310:
        import abc

        def build_class(self):
            """
            Finalize class based on the accumulated configuration.

            Builder cannot be used after calling this method.
            """
            if self._slots is True:
                return self._create_slots_class()
            return self.abc.update_abstractmethods(self._patch_original_class())
    else:

        def build_class(self):
            """
            Finalize class based on the accumulated configuration.

            Builder cannot be used after calling this method.
            """
            if self._slots is True:
                return self._create_slots_class()
            return self._patch_original_class()

    def _patch_original_class(self):
        """
        Apply accumulated methods and return the class.
        """
        cls = self._cls
        base_names = self._base_names
        if self._delete_attribs:
            for name in self._attr_names:
                if name not in base_names and getattr(cls, name, _sentinel) is not _sentinel:
                    with contextlib.suppress(AttributeError):
                        delattr(cls, name)
        for name, value in self._cls_dict.items():
            setattr(cls, name, value)
        if not self._wrote_own_setattr and getattr(cls, '__attrs_own_setattr__', False):
            cls.__attrs_own_setattr__ = False
            if not self._has_custom_setattr:
                cls.__setattr__ = _obj_setattr
        return cls

    def _create_slots_class(self):
        """
        Build and return a new class with a `__slots__` attribute.
        """
        cd = {k: v for k, v in self._cls_dict.items() if k not in (*tuple(self._attr_names), '__dict__', '__weakref__')}
        if not self._wrote_own_setattr:
            cd['__attrs_own_setattr__'] = False
            if not self._has_custom_setattr:
                for base_cls in self._cls.__bases__:
                    if base_cls.__dict__.get('__attrs_own_setattr__', False):
                        cd['__setattr__'] = _obj_setattr
                        break
        existing_slots = {}
        weakref_inherited = False
        for base_cls in self._cls.__mro__[1:-1]:
            if base_cls.__dict__.get('__weakref__', None) is not None:
                weakref_inherited = True
            existing_slots.update({name: getattr(base_cls, name) for name in getattr(base_cls, '__slots__', [])})
        base_names = set(self._base_names)
        names = self._attr_names
        if self._weakref_slot and '__weakref__' not in getattr(self._cls, '__slots__', ()) and ('__weakref__' not in names) and (not weakref_inherited):
            names += ('__weakref__',)
        if PY_3_8_PLUS:
            cached_properties = {name: cached_property.func for name, cached_property in cd.items() if isinstance(cached_property, functools.cached_property)}
        else:
            cached_properties = {}
        additional_closure_functions_to_update = []
        if cached_properties:
            names += tuple(cached_properties.keys())
            for name in cached_properties:
                del cd[name]
            class_annotations = _get_annotations(self._cls)
            for name, func in cached_properties.items():
                annotation = inspect.signature(func).return_annotation
                if annotation is not inspect.Parameter.empty:
                    class_annotations[name] = annotation
            original_getattr = cd.get('__getattr__')
            if original_getattr is not None:
                additional_closure_functions_to_update.append(original_getattr)
            cd['__getattr__'] = _make_cached_property_getattr(cached_properties, original_getattr, self._cls)
        slot_names = [name for name in names if name not in base_names]
        reused_slots = {slot: slot_descriptor for slot, slot_descriptor in existing_slots.items() if slot in slot_names}
        slot_names = [name for name in slot_names if name not in reused_slots]
        cd.update(reused_slots)
        if self._cache_hash:
            slot_names.append(_hash_cache_field)
        cd['__slots__'] = tuple(slot_names)
        cd['__qualname__'] = self._cls.__qualname__
        cls = type(self._cls)(self._cls.__name__, self._cls.__bases__, cd)
        for item in itertools.chain(cls.__dict__.values(), additional_closure_functions_to_update):
            if isinstance(item, (classmethod, staticmethod)):
                closure_cells = getattr(item.__func__, '__closure__', None)
            elif isinstance(item, property):
                closure_cells = getattr(item.fget, '__closure__', None)
            else:
                closure_cells = getattr(item, '__closure__', None)
            if not closure_cells:
                continue
            for cell in closure_cells:
                try:
                    match = cell.cell_contents is self._cls
                except ValueError:
                    pass
                else:
                    if match:
                        cell.cell_contents = cls
        return cls

    def add_repr(self, ns):
        self._cls_dict['__repr__'] = self._add_method_dunders(_make_repr(self._attrs, ns, self._cls))
        return self

    def add_str(self):
        repr = self._cls_dict.get('__repr__')
        if repr is None:
            msg = '__str__ can only be generated if a __repr__ exists.'
            raise ValueError(msg)

        def __str__(self):
            return self.__repr__()
        self._cls_dict['__str__'] = self._add_method_dunders(__str__)
        return self

    def _make_getstate_setstate(self):
        """
        Create custom __setstate__ and __getstate__ methods.
        """
        state_attr_names = tuple((an for an in self._attr_names if an != '__weakref__'))

        def slots_getstate(self):
            """
            Automatically created by attrs.
            """
            return {name: getattr(self, name) for name in state_attr_names}
        hash_caching_enabled = self._cache_hash

        def slots_setstate(self, state):
            """
            Automatically created by attrs.
            """
            __bound_setattr = _obj_setattr.__get__(self)
            if isinstance(state, tuple):
                for name, value in zip(state_attr_names, state):
                    __bound_setattr(name, value)
            else:
                for name in state_attr_names:
                    if name in state:
                        __bound_setattr(name, state[name])
            if hash_caching_enabled:
                __bound_setattr(_hash_cache_field, None)
        return (slots_getstate, slots_setstate)

    def make_unhashable(self):
        self._cls_dict['__hash__'] = None
        return self

    def add_hash(self):
        self._cls_dict['__hash__'] = self._add_method_dunders(_make_hash(self._cls, self._attrs, frozen=self._frozen, cache_hash=self._cache_hash))
        return self

    def add_init(self):
        self._cls_dict['__init__'] = self._add_method_dunders(_make_init(self._cls, self._attrs, self._has_pre_init, self._pre_init_has_args, self._has_post_init, self._frozen, self._slots, self._cache_hash, self._base_attr_map, self._is_exc, self._on_setattr, attrs_init=False))
        return self

    def add_match_args(self):
        self._cls_dict['__match_args__'] = tuple((field.name for field in self._attrs if field.init and (not field.kw_only)))

    def add_attrs_init(self):
        self._cls_dict['__attrs_init__'] = self._add_method_dunders(_make_init(self._cls, self._attrs, self._has_pre_init, self._pre_init_has_args, self._has_post_init, self._frozen, self._slots, self._cache_hash, self._base_attr_map, self._is_exc, self._on_setattr, attrs_init=True))
        return self

    def add_eq(self):
        cd = self._cls_dict
        cd['__eq__'] = self._add_method_dunders(_make_eq(self._cls, self._attrs))
        cd['__ne__'] = self._add_method_dunders(_make_ne())
        return self

    def add_order(self):
        cd = self._cls_dict
        cd['__lt__'], cd['__le__'], cd['__gt__'], cd['__ge__'] = (self._add_method_dunders(meth) for meth in _make_order(self._cls, self._attrs))
        return self

    def add_setattr(self):
        if self._frozen:
            return self
        sa_attrs = {}
        for a in self._attrs:
            on_setattr = a.on_setattr or self._on_setattr
            if on_setattr and on_setattr is not setters.NO_OP:
                sa_attrs[a.name] = (a, on_setattr)
        if not sa_attrs:
            return self
        if self._has_custom_setattr:
            msg = "Can't combine custom __setattr__ with on_setattr hooks."
            raise ValueError(msg)

        def __setattr__(self, name, val):
            try:
                a, hook = sa_attrs[name]
            except KeyError:
                nval = val
            else:
                nval = hook(self, a, val)
            _obj_setattr(self, name, nval)
        self._cls_dict['__attrs_own_setattr__'] = True
        self._cls_dict['__setattr__'] = self._add_method_dunders(__setattr__)
        self._wrote_own_setattr = True
        return self

    def _add_method_dunders(self, method):
        """
        Add __module__ and __qualname__ to a *method* if possible.
        """
        with contextlib.suppress(AttributeError):
            method.__module__ = self._cls.__module__
        with contextlib.suppress(AttributeError):
            method.__qualname__ = f'{self._cls.__qualname__}.{method.__name__}'
        with contextlib.suppress(AttributeError):
            method.__doc__ = f'Method generated by attrs for class {self._cls.__qualname__}.'
        return method