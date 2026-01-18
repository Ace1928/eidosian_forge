from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
def _AddInitMethod(message_descriptor, cls):
    """Adds an __init__ method to cls."""

    def _GetIntegerEnumValue(enum_type, value):
        """Convert a string or integer enum value to an integer.

    If the value is a string, it is converted to the enum value in
    enum_type with the same name.  If the value is not a string, it's
    returned as-is.  (No conversion or bounds-checking is done.)
    """
        if isinstance(value, str):
            try:
                return enum_type.values_by_name[value].number
            except KeyError:
                raise ValueError('Enum type %s: unknown label "%s"' % (enum_type.full_name, value))
        return value

    def init(self, **kwargs):
        self._cached_byte_size = 0
        self._cached_byte_size_dirty = len(kwargs) > 0
        self._fields = {}
        self._oneofs = {}
        self._unknown_fields = ()
        self._unknown_field_set = None
        self._is_present_in_parent = False
        self._listener = message_listener_mod.NullMessageListener()
        self._listener_for_children = _Listener(self)
        for field_name, field_value in kwargs.items():
            field = _GetFieldByName(message_descriptor, field_name)
            if field is None:
                raise TypeError('%s() got an unexpected keyword argument "%s"' % (message_descriptor.name, field_name))
            if field_value is None:
                continue
            if field.label == _FieldDescriptor.LABEL_REPEATED:
                copy = field._default_constructor(self)
                if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
                    if _IsMapField(field):
                        if _IsMessageMapField(field):
                            for key in field_value:
                                copy[key].MergeFrom(field_value[key])
                        else:
                            copy.update(field_value)
                    else:
                        for val in field_value:
                            if isinstance(val, dict):
                                copy.add(**val)
                            else:
                                copy.add().MergeFrom(val)
                else:
                    if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
                        field_value = [_GetIntegerEnumValue(field.enum_type, val) for val in field_value]
                    copy.extend(field_value)
                self._fields[field] = copy
            elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
                copy = field._default_constructor(self)
                new_val = field_value
                if isinstance(field_value, dict):
                    new_val = field.message_type._concrete_class(**field_value)
                try:
                    copy.MergeFrom(new_val)
                except TypeError:
                    _ReraiseTypeErrorWithFieldName(message_descriptor.name, field_name)
                self._fields[field] = copy
            else:
                if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
                    field_value = _GetIntegerEnumValue(field.enum_type, field_value)
                try:
                    setattr(self, field_name, field_value)
                except TypeError:
                    _ReraiseTypeErrorWithFieldName(message_descriptor.name, field_name)
    init.__module__ = None
    init.__doc__ = None
    cls.__init__ = init