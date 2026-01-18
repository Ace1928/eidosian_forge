from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersistenceModeValueValuesEnum(_messages.Enum):
    """Optional. Controls whether Persistence features are enabled. If not
    provided, the existing value will be used.

    Values:
      PERSISTENCE_MODE_UNSPECIFIED: Not set.
      DISABLED: Persistence is disabled for the instance, and any existing
        snapshots are deleted.
      RDB: RDB based Persistence is enabled.
    """
    PERSISTENCE_MODE_UNSPECIFIED = 0
    DISABLED = 1
    RDB = 2