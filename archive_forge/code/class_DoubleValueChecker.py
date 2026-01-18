import ctypes
import numbers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import wire_format
from google.protobuf import descriptor
class DoubleValueChecker(object):
    """Checker used for double fields.

  Performs type-check and range check.
  """

    def CheckValue(self, proposed_value):
        """Check and convert proposed_value to float."""
        if not hasattr(proposed_value, '__float__') and (not hasattr(proposed_value, '__index__')) or (type(proposed_value).__module__ == 'numpy' and type(proposed_value).__name__ == 'ndarray'):
            message = '%.1024r has type %s, but expected one of: int, float' % (proposed_value, type(proposed_value))
            raise TypeError(message)
        return float(proposed_value)

    def DefaultValue(self):
        return 0.0