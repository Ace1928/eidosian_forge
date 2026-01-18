from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationStatusValueValuesEnum(_messages.Enum):
    """Optional. Output only. Replication status of configured replication.

    Values:
      REPLICATION_STATUS_UNSPECIFIED: Default value.
      ACTIVE: Replication is Active with no errors.
      SOURCE_DELETED: Source object is deleted.
      PERMISSION_DENIED: Source revoked replication permissions.
      UNSUPPORTED_CONFIGURATION: Source configuration doesn't allow
        replication.
    """
    REPLICATION_STATUS_UNSPECIFIED = 0
    ACTIVE = 1
    SOURCE_DELETED = 2
    PERMISSION_DENIED = 3
    UNSUPPORTED_CONFIGURATION = 4