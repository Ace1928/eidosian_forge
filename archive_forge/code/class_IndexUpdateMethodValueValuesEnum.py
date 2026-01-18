from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexUpdateMethodValueValuesEnum(_messages.Enum):
    """Immutable. The update method to use with this Index. If not set,
    BATCH_UPDATE will be used by default.

    Values:
      INDEX_UPDATE_METHOD_UNSPECIFIED: Should not be used.
      BATCH_UPDATE: BatchUpdate: user can call UpdateIndex with files on Cloud
        Storage of Datapoints to update.
      STREAM_UPDATE: StreamUpdate: user can call
        UpsertDatapoints/DeleteDatapoints to update the Index and the updates
        will be applied in corresponding DeployedIndexes in nearly real-time.
    """
    INDEX_UPDATE_METHOD_UNSPECIFIED = 0
    BATCH_UPDATE = 1
    STREAM_UPDATE = 2