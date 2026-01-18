import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class _proto_member:
    """
    intermediate step for enum members between class execution and final creation
    """

    def __init__(self, value):
        self.value = value

    def __set_name__(self, enum_class, member_name):
        """
        convert each quasi-member into an instance of the new enum class
        """
        delattr(enum_class, member_name)
        value = self.value
        if not isinstance(value, tuple):
            args = (value,)
        else:
            args = value
        if enum_class._member_type_ is tuple:
            args = (args,)
        if not enum_class._use_args_:
            enum_member = enum_class._new_member_(enum_class)
        else:
            enum_member = enum_class._new_member_(enum_class, *args)
        if not hasattr(enum_member, '_value_'):
            if enum_class._member_type_ is object:
                enum_member._value_ = value
            else:
                try:
                    enum_member._value_ = enum_class._member_type_(*args)
                except Exception as exc:
                    new_exc = TypeError('_value_ not set in __new__, unable to create it')
                    new_exc.__cause__ = exc
                    raise new_exc
        value = enum_member._value_
        enum_member._name_ = member_name
        enum_member.__objclass__ = enum_class
        enum_member.__init__(*args)
        enum_member._sort_order_ = len(enum_class._member_names_)
        if Flag is not None and issubclass(enum_class, Flag):
            enum_class._flag_mask_ |= value
            if _is_single_bit(value):
                enum_class._singles_mask_ |= value
            enum_class._all_bits_ = 2 ** enum_class._flag_mask_.bit_length() - 1
        try:
            try:
                enum_member = enum_class._value2member_map_[value]
            except TypeError:
                for name, canonical_member in enum_class._member_map_.items():
                    if canonical_member._value_ == value:
                        enum_member = canonical_member
                        break
                else:
                    raise KeyError
        except KeyError:
            if Flag is None or not issubclass(enum_class, Flag):
                enum_class._member_names_.append(member_name)
            elif Flag is not None and issubclass(enum_class, Flag) and _is_single_bit(value):
                enum_class._member_names_.append(member_name)
        found_descriptor = None
        for base in enum_class.__mro__[1:]:
            descriptor = base.__dict__.get(member_name)
            if descriptor is not None:
                if isinstance(descriptor, (property, DynamicClassAttribute)):
                    found_descriptor = descriptor
                    break
                elif hasattr(descriptor, 'fget') and hasattr(descriptor, 'fset') and hasattr(descriptor, 'fdel'):
                    found_descriptor = descriptor
                    continue
        if found_descriptor:
            redirect = property()
            redirect.member = enum_member
            redirect.__set_name__(enum_class, member_name)
            redirect.fget = found_descriptor.fget
            redirect.fset = found_descriptor.fset
            redirect.fdel = found_descriptor.fdel
            setattr(enum_class, member_name, redirect)
        else:
            setattr(enum_class, member_name, enum_member)
        enum_class._member_map_[member_name] = enum_member
        try:
            enum_class._value2member_map_.setdefault(value, enum_member)
        except TypeError:
            enum_class._unhashable_values_.append(value)