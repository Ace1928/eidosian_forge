import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _simple_enum(etype=Enum, *, boundary=None, use_args=None):
    """
    Class decorator that converts a normal class into an :class:`Enum`.  No
    safety checks are done, and some advanced behavior (such as
    :func:`__init_subclass__`) is not available.  Enum creation can be faster
    using :func:`simple_enum`.

        >>> from enum import Enum, _simple_enum
        >>> @_simple_enum(Enum)
        ... class Color:
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> Color
        <enum 'Color'>
    """

    def convert_class(cls):
        nonlocal use_args
        cls_name = cls.__name__
        if use_args is None:
            use_args = etype._use_args_
        __new__ = cls.__dict__.get('__new__')
        if __new__ is not None:
            new_member = __new__.__func__
        else:
            new_member = etype._member_type_.__new__
        attrs = {}
        body = {}
        if __new__ is not None:
            body['__new_member__'] = new_member
        body['_new_member_'] = new_member
        body['_use_args_'] = use_args
        body['_generate_next_value_'] = gnv = etype._generate_next_value_
        body['_member_names_'] = member_names = []
        body['_member_map_'] = member_map = {}
        body['_value2member_map_'] = value2member_map = {}
        body['_unhashable_values_'] = []
        body['_member_type_'] = member_type = etype._member_type_
        body['_value_repr_'] = etype._value_repr_
        if issubclass(etype, Flag):
            body['_boundary_'] = boundary or etype._boundary_
            body['_flag_mask_'] = None
            body['_all_bits_'] = None
            body['_singles_mask_'] = None
            body['_inverted_'] = None
            body['__or__'] = Flag.__or__
            body['__xor__'] = Flag.__xor__
            body['__and__'] = Flag.__and__
            body['__ror__'] = Flag.__ror__
            body['__rxor__'] = Flag.__rxor__
            body['__rand__'] = Flag.__rand__
            body['__invert__'] = Flag.__invert__
        for name, obj in cls.__dict__.items():
            if name in ('__dict__', '__weakref__'):
                continue
            if _is_dunder(name) or _is_private(cls_name, name) or _is_sunder(name) or _is_descriptor(obj):
                body[name] = obj
            else:
                attrs[name] = obj
        if cls.__dict__.get('__doc__') is None:
            body['__doc__'] = 'An enumeration.'
        enum_class = type(cls_name, (etype,), body, boundary=boundary, _simple=True)
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name not in body:
                enum_method = getattr(etype, name)
                found_method = getattr(enum_class, name)
                object_method = getattr(object, name)
                data_type_method = getattr(member_type, name)
                if found_method in (data_type_method, object_method):
                    setattr(enum_class, name, enum_method)
        gnv_last_values = []
        if issubclass(enum_class, Flag):
            single_bits = multi_bits = 0
            for name, value in attrs.items():
                if isinstance(value, auto) and auto.value is _auto_null:
                    value = gnv(name, 1, len(member_names), gnv_last_values)
                if value in value2member_map:
                    redirect = property()
                    redirect.__set_name__(enum_class, name)
                    setattr(enum_class, name, redirect)
                    member_map[name] = value2member_map[value]
                else:
                    if use_args:
                        if not isinstance(value, tuple):
                            value = (value,)
                        member = new_member(enum_class, *value)
                        value = value[0]
                    else:
                        member = new_member(enum_class)
                    if __new__ is None:
                        member._value_ = value
                    member._name_ = name
                    member.__objclass__ = enum_class
                    member.__init__(value)
                    redirect = property()
                    redirect.__set_name__(enum_class, name)
                    setattr(enum_class, name, redirect)
                    member_map[name] = member
                    member._sort_order_ = len(member_names)
                    value2member_map[value] = member
                    if _is_single_bit(value):
                        member_names.append(name)
                        single_bits |= value
                    else:
                        multi_bits |= value
                    gnv_last_values.append(value)
            enum_class._flag_mask_ = single_bits | multi_bits
            enum_class._singles_mask_ = single_bits
            enum_class._all_bits_ = 2 ** (single_bits | multi_bits).bit_length() - 1
            member_list = [m._value_ for m in enum_class]
            if member_list != sorted(member_list):
                enum_class._iter_member_ = enum_class._iter_member_by_def_
        else:
            for name, value in attrs.items():
                if isinstance(value, auto):
                    if value.value is _auto_null:
                        value.value = gnv(name, 1, len(member_names), gnv_last_values)
                    value = value.value
                if value in value2member_map:
                    redirect = property()
                    redirect.__set_name__(enum_class, name)
                    setattr(enum_class, name, redirect)
                    member_map[name] = value2member_map[value]
                else:
                    if use_args:
                        if not isinstance(value, tuple):
                            value = (value,)
                        member = new_member(enum_class, *value)
                        value = value[0]
                    else:
                        member = new_member(enum_class)
                    if __new__ is None:
                        member._value_ = value
                    member._name_ = name
                    member.__objclass__ = enum_class
                    member.__init__(value)
                    member._sort_order_ = len(member_names)
                    redirect = property()
                    redirect.__set_name__(enum_class, name)
                    setattr(enum_class, name, redirect)
                    member_map[name] = member
                    value2member_map[value] = member
                    member_names.append(name)
                    gnv_last_values.append(value)
        if '__new__' in body:
            enum_class.__new_member__ = enum_class.__new__
        enum_class.__new__ = Enum.__new__
        return enum_class
    return convert_class