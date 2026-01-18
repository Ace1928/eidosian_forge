from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FetchFeatureValuesRequest(_messages.Message):
    """Request message for FeatureOnlineStoreService.FetchFeatureValues. All
  the features under the requested feature view will be returned.

  Enums:
    DataFormatValueValuesEnum: Optional. Response data format. If not set,
      FeatureViewDataFormat.KEY_VALUE will be used.
    FormatValueValuesEnum: Specify response data format. If not set, KeyValue
      format will be used. Deprecated. Use
      FetchFeatureValuesRequest.data_format.

  Fields:
    dataFormat: Optional. Response data format. If not set,
      FeatureViewDataFormat.KEY_VALUE will be used.
    dataKey: Optional. The request key to fetch feature values for.
    format: Specify response data format. If not set, KeyValue format will be
      used. Deprecated. Use FetchFeatureValuesRequest.data_format.
    id: Simple ID. The whole string will be used as is to identify Entity to
      fetch feature values for.
  """

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
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 1)
    dataKey = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewDataKey', 2)
    format = _messages.EnumField('FormatValueValuesEnum', 3)
    id = _messages.StringField(4)