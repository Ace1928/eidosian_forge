from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepairNodeGroupRequest(_messages.Message):
    """A RepairNodeGroupRequest object.

  Enums:
    RepairActionValueValuesEnum: Required. Repair action to take on specified
      resources of the node pool.

  Fields:
    instanceNames: Required. Name of instances to be repaired. These instances
      must belong to specified node pool.
    repairAction: Required. Repair action to take on specified resources of
      the node pool.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two RepairNodeGroupRequest with the same ID, the second
      request is ignored and the first google.longrunning.Operation created
      and stored in the backend is returned.Recommendation: Set this value to
      a UUID (https://en.wikipedia.org/wiki/Universally_unique_identifier).The
      ID must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """

    class RepairActionValueValuesEnum(_messages.Enum):
        """Required. Repair action to take on specified resources of the node
    pool.

    Values:
      REPAIR_ACTION_UNSPECIFIED: No action will be taken by default.
      REPLACE: replace the specified list of nodes.
    """
        REPAIR_ACTION_UNSPECIFIED = 0
        REPLACE = 1
    instanceNames = _messages.StringField(1, repeated=True)
    repairAction = _messages.EnumField('RepairActionValueValuesEnum', 2)
    requestId = _messages.StringField(3)