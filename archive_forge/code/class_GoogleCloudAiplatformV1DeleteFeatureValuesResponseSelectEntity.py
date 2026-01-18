from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DeleteFeatureValuesResponseSelectEntity(_messages.Message):
    """Response message if the request uses the SelectEntity option.

  Fields:
    offlineStorageDeletedEntityRowCount: The count of deleted entity rows in
      the offline storage. Each row corresponds to the combination of an
      entity ID and a timestamp. One entity ID can have multiple rows in the
      offline storage.
    onlineStorageDeletedEntityCount: The count of deleted entities in the
      online storage. Each entity ID corresponds to one entity.
  """
    offlineStorageDeletedEntityRowCount = _messages.IntegerField(1)
    onlineStorageDeletedEntityCount = _messages.IntegerField(2)