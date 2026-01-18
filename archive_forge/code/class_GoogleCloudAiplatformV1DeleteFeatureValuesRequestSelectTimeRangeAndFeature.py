from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeleteFeatureValuesRequestSelectTimeRangeAndFeature(_messages.Message):
    """Message to select time range and feature. Values of the selected feature
  generated within an inclusive time range will be deleted. Using this option
  permanently deletes the feature values from the specified feature IDs within
  the specified time range. This might include data from the online storage.
  If you want to retain any deleted historical data in the online storage, you
  must re-ingest it.

  Fields:
    featureSelector: Required. Selectors choosing which feature values to be
      deleted from the EntityType.
    skipOnlineStorageDelete: If set, data will not be deleted from online
      storage. When time range is older than the data in online storage,
      setting this to be true will make the deletion have no impact on online
      serving.
    timeRange: Required. Select feature generated within a half-inclusive time
      range. The time range is lower inclusive and upper exclusive.
  """
    featureSelector = _messages.MessageField('GoogleCloudAiplatformV1FeatureSelector', 1)
    skipOnlineStorageDelete = _messages.BooleanField(2)
    timeRange = _messages.MessageField('GoogleTypeInterval', 3)