from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FormatValueValuesEnum(_messages.Enum):
    """Specify response data format. If not set, KeyValue format will be
    used. Deprecated. Use FetchFeatureValuesRequest.data_format.

    Values:
      FORMAT_UNSPECIFIED: Not set. Will be treated as the KeyValue format.
      KEY_VALUE: Return response data in key-value format.
      PROTO_STRUCT: Return response data in proto Struct format.
    """
    FORMAT_UNSPECIFIED = 0
    KEY_VALUE = 1
    PROTO_STRUCT = 2