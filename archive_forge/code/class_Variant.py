import types
import weakref
import six
from apitools.base.protorpclite import util
class Variant(Enum):
    """Wire format variant.

    Used by the 'protobuf' wire format to determine how to transmit
    a single piece of data.  May be used by other formats.

    See: http://code.google.com/apis/protocolbuffers/docs/encoding.html

    Values:
      DOUBLE: 64-bit floating point number.
      FLOAT: 32-bit floating point number.
      INT64: 64-bit signed integer.
      UINT64: 64-bit unsigned integer.
      INT32: 32-bit signed integer.
      BOOL: Boolean value (True or False).
      STRING: String of UTF-8 encoded text.
      MESSAGE: Embedded message as byte string.
      BYTES: String of 8-bit bytes.
      UINT32: 32-bit unsigned integer.
      ENUM: Enum value as integer.
      SINT32: 32-bit signed integer.  Uses "zig-zag" encoding.
      SINT64: 64-bit signed integer.  Uses "zig-zag" encoding.
    """
    DOUBLE = 1
    FLOAT = 2
    INT64 = 3
    UINT64 = 4
    INT32 = 5
    BOOL = 8
    STRING = 9
    MESSAGE = 11
    BYTES = 12
    UINT32 = 13
    ENUM = 14
    SINT32 = 17
    SINT64 = 18