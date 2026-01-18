from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskResourceStatusAsyncReplicationStatus(_messages.Message):
    """A DiskResourceStatusAsyncReplicationStatus object.

  Enums:
    StateValueValuesEnum:

  Fields:
    state: A StateValueValuesEnum attribute.
  """

    class StateValueValuesEnum(_messages.Enum):
        """StateValueValuesEnum enum type.

    Values:
      ACTIVE: Replication is active.
      CREATED: Secondary disk is created and is waiting for replication to
        start.
      STARTING: Replication is starting.
      STATE_UNSPECIFIED: <no description>
      STOPPED: Replication is stopped.
      STOPPING: Replication is stopping.
    """
        ACTIVE = 0
        CREATED = 1
        STARTING = 2
        STATE_UNSPECIFIED = 3
        STOPPED = 4
        STOPPING = 5
    state = _messages.EnumField('StateValueValuesEnum', 1)