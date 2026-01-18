from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBytesStatusValueValuesEnum(_messages.Enum):
    """[Output Only] An indicator whether storageBytes is in a stable state
    or it is being adjusted as a result of shared storage reallocation. This
    status can either be UPDATING, meaning the size of the snapshot is being
    updated, or UP_TO_DATE, meaning the size of the snapshot is up-to-date.

    Values:
      UPDATING: <no description>
      UP_TO_DATE: <no description>
    """
    UPDATING = 0
    UP_TO_DATE = 1