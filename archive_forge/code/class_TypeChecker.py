import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class TypeChecker(object):
    """Type checker used to catch type errors as early as possible
  when the client is setting scalar fields in protocol messages.
  """

    def __init__(self, *acceptable_types):
        self._acceptable_types = acceptable_types

    def CheckValue(self, proposed_value):
        """Type check the provided value and return it.

    The returned value might have been normalized to another type.
    """
        if not isinstance(proposed_value, self._acceptable_types):
            message = '%.1024r has type %s, but expected one of: %s' % (proposed_value, type(proposed_value), self._acceptable_types)
            raise TypeError(message)
        if self._acceptable_types:
            if self._acceptable_types[0] in (bool, float):
                return self._acceptable_types[0](proposed_value)
        return proposed_value