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
def _make_init(cls, attrs, pre_init, pre_init_has_args, post_init, frozen, slots, cache_hash, base_attr_map, is_exc, cls_on_setattr, attrs_init):
    has_cls_on_setattr = cls_on_setattr is not None and cls_on_setattr is not setters.NO_OP
    if frozen and has_cls_on_setattr:
        msg = "Frozen classes can't use on_setattr."
        raise ValueError(msg)
    needs_cached_setattr = cache_hash or frozen
    filtered_attrs = []
    attr_dict = {}
    for a in attrs:
        if not a.init and a.default is NOTHING:
            continue
        filtered_attrs.append(a)
        attr_dict[a.name] = a
        if a.on_setattr is not None:
            if frozen is True:
                msg = "Frozen classes can't use on_setattr."
                raise ValueError(msg)
            needs_cached_setattr = True
        elif has_cls_on_setattr and a.on_setattr is not setters.NO_OP:
            needs_cached_setattr = True
    unique_filename = _generate_unique_filename(cls, 'init')
    script, globs, annotations = _attrs_to_init_script(filtered_attrs, frozen, slots, pre_init, pre_init_has_args, post_init, cache_hash, base_attr_map, is_exc, needs_cached_setattr, has_cls_on_setattr, attrs_init)
    if cls.__module__ in sys.modules:
        globs.update(sys.modules[cls.__module__].__dict__)
    globs.update({'NOTHING': NOTHING, 'attr_dict': attr_dict})
    if needs_cached_setattr:
        globs['_cached_setattr_get'] = _obj_setattr.__get__
    init = _make_method('__attrs_init__' if attrs_init else '__init__', script, unique_filename, globs)
    init.__annotations__ = annotations
    return init