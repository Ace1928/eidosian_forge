from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolesValueListEntryValuesEnum(_messages.Enum):
    """RolesValueListEntryValuesEnum enum type.

    Values:
      ROLE_UNSPECIFIED: Required unspecified role.
      DRIVER: Job drivers run on the node pool.
      MASTER: Master nodes.
      PRIMARY_WORKER: Primary workers.
      SECONDARY_WORKER: Secondary workers.
    """
    ROLE_UNSPECIFIED = 0
    DRIVER = 1
    MASTER = 2
    PRIMARY_WORKER = 3
    SECONDARY_WORKER = 4