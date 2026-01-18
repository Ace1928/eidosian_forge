import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _get_field(cls, a_name, a_type, default_kw_only):
    default = getattr(cls, a_name, MISSING)
    if isinstance(default, Field):
        f = default
    else:
        if isinstance(default, types.MemberDescriptorType):
            default = MISSING
        f = field(default=default)
    f.name = a_name
    f.type = a_type
    f._field_type = _FIELD
    typing = sys.modules.get('typing')
    if typing:
        if _is_classvar(a_type, typing) or (isinstance(f.type, str) and _is_type(f.type, cls, typing, typing.ClassVar, _is_classvar)):
            f._field_type = _FIELD_CLASSVAR
    if f._field_type is _FIELD:
        dataclasses = sys.modules[__name__]
        if _is_initvar(a_type, dataclasses) or (isinstance(f.type, str) and _is_type(f.type, cls, dataclasses, dataclasses.InitVar, _is_initvar)):
            f._field_type = _FIELD_INITVAR
    if f._field_type in (_FIELD_CLASSVAR, _FIELD_INITVAR):
        if f.default_factory is not MISSING:
            raise TypeError(f'field {f.name} cannot have a default factory')
    if f._field_type in (_FIELD, _FIELD_INITVAR):
        if f.kw_only is MISSING:
            f.kw_only = default_kw_only
    else:
        assert f._field_type is _FIELD_CLASSVAR
        if f.kw_only is not MISSING:
            raise TypeError(f'field {f.name} is a ClassVar but specifies kw_only')
    if f._field_type is _FIELD and f.default.__class__.__hash__ is None:
        raise ValueError(f'mutable default {type(f.default)} for field {f.name} is not allowed: use default_factory')
    return f