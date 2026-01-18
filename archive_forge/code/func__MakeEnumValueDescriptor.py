import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _MakeEnumValueDescriptor(self, value_proto, index):
    """Creates a enum value descriptor object from a enum value proto.

    Args:
      value_proto: The proto describing the enum value.
      index: The index of the enum value.

    Returns:
      An initialized EnumValueDescriptor object.
    """
    return descriptor.EnumValueDescriptor(name=value_proto.name, index=index, number=value_proto.number, options=_OptionsOrNone(value_proto), type=None, create_key=descriptor._internal_create_key)