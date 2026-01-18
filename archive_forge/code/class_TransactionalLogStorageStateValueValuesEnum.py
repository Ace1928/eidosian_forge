from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TransactionalLogStorageStateValueValuesEnum(_messages.Enum):
    """Output only. This value contains the storage location of transactional
    logs for the database for point-in-time recovery.

    Values:
      TRANSACTIONAL_LOG_STORAGE_STATE_UNSPECIFIED: Unspecified.
      DISK: The transaction logs used for PITR for the instance are stored on
        a data disk.
      SWITCHING_TO_CLOUD_STORAGE: The transaction logs used for PITR for the
        instance are switching from being stored on a data disk to being
        stored in Cloud Storage. Only applicable to MySQL.
      SWITCHED_TO_CLOUD_STORAGE: The transaction logs used for PITR for the
        instance are now stored in Cloud Storage. Previously, they were stored
        on a data disk. Only applicable to MySQL.
      CLOUD_STORAGE: The transaction logs used for PITR for the instance are
        stored in Cloud Storage. Only applicable to MySQL and PostgreSQL.
    """
    TRANSACTIONAL_LOG_STORAGE_STATE_UNSPECIFIED = 0
    DISK = 1
    SWITCHING_TO_CLOUD_STORAGE = 2
    SWITCHED_TO_CLOUD_STORAGE = 3
    CLOUD_STORAGE = 4