import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class FloatValueChecker(object):
    """Checker used for float fields.  Performs type-check and range check.

  Values exceeding a 32-bit float will be converted to inf/-inf.
  """

    def CheckValue(self, proposed_value):
        """Check and convert proposed_value to float."""
        if not isinstance(proposed_value, numbers.Real):
            message = '%.1024r has type %s, but expected one of: numbers.Real' % (proposed_value, type(proposed_value))
            raise TypeError(message)
        converted_value = float(proposed_value)
        if converted_value > _FLOAT_MAX:
            return _INF
        if converted_value < _FLOAT_MIN:
            return _NEG_INF
        return TruncateToFourByteFloat(converted_value)

    def DefaultValue(self):
        return 0.0