import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
def GetTypeChecker(field):
    """Returns a type checker for a message field of the specified types.

  Args:
    field: FieldDescriptor object for this field.

  Returns:
    An instance of TypeChecker which can be used to verify the types
    of values assigned to a field of the specified type.
  """
    if field.cpp_type == _FieldDescriptor.CPPTYPE_STRING and field.type == _FieldDescriptor.TYPE_STRING:
        return UnicodeValueChecker()
    if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
        if SupportsOpenEnums(field):
            return _VALUE_CHECKERS[_FieldDescriptor.CPPTYPE_INT32]
        else:
            return EnumValueChecker(field.enum_type)
    return _VALUE_CHECKERS[field.cpp_type]