import struct
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import message
def IsTypePackable(field_type):
    """Return true iff packable = true is valid for fields of this type.

  Args:
    field_type: a FieldDescriptor::Type value.

  Returns:
    True iff fields of this type are packable.
  """
    return field_type not in NON_PACKABLE_TYPES