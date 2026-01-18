import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def EnumValueName(self, enum, value):
    """Returns the string name of an enum value.

    This is just a small helper method to simplify a common operation.

    Args:
      enum: string name of the Enum.
      value: int, value of the enum.

    Returns:
      string name of the enum value.

    Raises:
      KeyError if either the Enum doesn't exist or the value is not a valid
        value for the enum.
    """
    return self.enum_types_by_name[enum].values_by_number[value].name