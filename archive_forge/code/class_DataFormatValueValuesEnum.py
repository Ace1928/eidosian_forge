from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataFormatValueValuesEnum(_messages.Enum):
    """Optional. Response data format. If not set,
    FeatureViewDataFormat.KEY_VALUE will be used.

    Values:
      FEATURE_VIEW_DATA_FORMAT_UNSPECIFIED: Not set. Will be treated as the
        KeyValue format.
      KEY_VALUE: Return response data in key-value format.
      PROTO_STRUCT: Return response data in proto Struct format.
    """
    FEATURE_VIEW_DATA_FORMAT_UNSPECIFIED = 0
    KEY_VALUE = 1
    PROTO_STRUCT = 2