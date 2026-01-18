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
def _DefaultValueConstructorForField(field):
    """Returns a function which returns a default value for a field.

  Args:
    field: FieldDescriptor object for this field.

  The returned function has one argument:
    message: Message instance containing this field, or a weakref proxy
      of same.

  That function in turn returns a default value for this field.  The default
    value may refer back to |message| via a weak reference.
  """
    if _IsMapField(field):
        return _GetInitializeDefaultForMap(field)
    if field.label == _FieldDescriptor.LABEL_REPEATED:
        if field.has_default_value and field.default_value != []:
            raise ValueError('Repeated field default value not empty list: %s' % field.default_value)
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
            message_type = field.message_type

            def MakeRepeatedMessageDefault(message):
                return containers.RepeatedCompositeFieldContainer(message._listener_for_children, field.message_type)
            return MakeRepeatedMessageDefault
        else:
            type_checker = type_checkers.GetTypeChecker(field)

            def MakeRepeatedScalarDefault(message):
                return containers.RepeatedScalarFieldContainer(message._listener_for_children, type_checker)
            return MakeRepeatedScalarDefault
    if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        message_type = field.message_type

        def MakeSubMessageDefault(message):
            if not hasattr(message_type, '_concrete_class'):
                from google.protobuf import message_factory
                message_factory.GetMessageClass(message_type)
            result = message_type._concrete_class()
            result._SetListener(_OneofListener(message, field) if field.containing_oneof is not None else message._listener_for_children)
            return result
        return MakeSubMessageDefault

    def MakeScalarDefault(message):
        return field.default_value
    return MakeScalarDefault