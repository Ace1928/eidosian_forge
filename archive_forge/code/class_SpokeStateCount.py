from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpokeStateCount(_messages.Message):
    """The number of spokes that are in a particular state and associated with
  a given hub.

  Enums:
    StateValueValuesEnum: Output only. The state of the spokes.

  Fields:
    count: Output only. The total number of spokes that are in this state and
      associated with a given hub.
    state: Output only. The state of the spokes.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the spokes.

    Values:
      STATE_UNSPECIFIED: No state information available
      CREATING: The resource's create operation is in progress.
      ACTIVE: The resource is active
      DELETING: The resource's delete operation is in progress.
      ACTIVATING: The resource's activate operation is in progress.
      DEACTIVATING: The resource's deactivate operation is in progress.
      ACCEPTING: The resource's accept operation is in progress.
      REJECTING: The resource's reject operation is in progress.
      UPDATING: The resource's update operation is in progress.
      INACTIVE: The resource is inactive.
      OBSOLETE: The hub associated with this spoke resource has been deleted.
        This state applies to spoke resources only.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        ACTIVATING = 4
        DEACTIVATING = 5
        ACCEPTING = 6
        REJECTING = 7
        UPDATING = 8
        INACTIVE = 9
        OBSOLETE = 10
    count = _messages.IntegerField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)