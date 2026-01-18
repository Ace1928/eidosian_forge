import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class EnumValueChecker(object):
    """Checker used for enum fields.  Performs type-check and range check."""

    def __init__(self, enum_type):
        self._enum_type = enum_type

    def CheckValue(self, proposed_value):
        if not isinstance(proposed_value, numbers.Integral):
            message = '%.1024r has type %s, but expected one of: %s' % (proposed_value, type(proposed_value), (int,))
            raise TypeError(message)
        if int(proposed_value) not in self._enum_type.values_by_number:
            raise ValueError('Unknown enum value: %d' % proposed_value)
        return proposed_value

    def DefaultValue(self):
        return self._enum_type.values[0].number