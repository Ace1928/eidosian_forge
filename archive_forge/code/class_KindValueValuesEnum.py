from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KindValueValuesEnum(_messages.Enum):
    """The field type.

    Values:
      TYPE_UNKNOWN: Field type unknown.
      TYPE_DOUBLE: Field type double.
      TYPE_FLOAT: Field type float.
      TYPE_INT64: Field type int64.
      TYPE_UINT64: Field type uint64.
      TYPE_INT32: Field type int32.
      TYPE_FIXED64: Field type fixed64.
      TYPE_FIXED32: Field type fixed32.
      TYPE_BOOL: Field type bool.
      TYPE_STRING: Field type string.
      TYPE_GROUP: Field type group. Proto2 syntax only, and deprecated.
      TYPE_MESSAGE: Field type message.
      TYPE_BYTES: Field type bytes.
      TYPE_UINT32: Field type uint32.
      TYPE_ENUM: Field type enum.
      TYPE_SFIXED32: Field type sfixed32.
      TYPE_SFIXED64: Field type sfixed64.
      TYPE_SINT32: Field type sint32.
      TYPE_SINT64: Field type sint64.
    """
    TYPE_UNKNOWN = 0
    TYPE_DOUBLE = 1
    TYPE_FLOAT = 2
    TYPE_INT64 = 3
    TYPE_UINT64 = 4
    TYPE_INT32 = 5
    TYPE_FIXED64 = 6
    TYPE_FIXED32 = 7
    TYPE_BOOL = 8
    TYPE_STRING = 9
    TYPE_GROUP = 10
    TYPE_MESSAGE = 11
    TYPE_BYTES = 12
    TYPE_UINT32 = 13
    TYPE_ENUM = 14
    TYPE_SFIXED32 = 15
    TYPE_SFIXED64 = 16
    TYPE_SINT32 = 17
    TYPE_SINT64 = 18