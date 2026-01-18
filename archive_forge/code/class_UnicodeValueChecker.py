import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class UnicodeValueChecker(object):
    """Checker used for string fields.

  Always returns a unicode value, even if the input is of type str.
  """

    def CheckValue(self, proposed_value):
        if not isinstance(proposed_value, (bytes, str)):
            message = '%.1024r has type %s, but expected one of: %s' % (proposed_value, type(proposed_value), (bytes, str))
            raise TypeError(message)
        if isinstance(proposed_value, bytes):
            try:
                proposed_value = proposed_value.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError("%.1024r has type bytes, but isn't valid UTF-8 encoding. Non-UTF-8 strings must be converted to unicode objects before being added." % proposed_value)
        else:
            try:
                proposed_value.encode('utf8')
            except UnicodeEncodeError:
                raise ValueError("%.1024r isn't a valid unicode string and can't be encoded in UTF-8." % proposed_value)
        return proposed_value

    def DefaultValue(self):
        return u''