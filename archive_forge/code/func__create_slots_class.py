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