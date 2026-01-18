from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1p1beta1RunAssetDiscoveryResponse(_messages.Message):
    """Response of asset discovery run

  Enums:
    StateValueValuesEnum: The state of an asset discovery run.

  Fields:
    duration: The duration between asset discovery run start and end
    state: The state of an asset discovery run.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of an asset discovery run.

    Values:
      STATE_UNSPECIFIED: Asset discovery run state was unspecified.
      COMPLETED: Asset discovery run completed successfully.
      SUPERSEDED: Asset discovery run was cancelled with tasks still pending,
        as another run for the same organization was started with a higher
        priority.
      TERMINATED: Asset discovery run was killed and terminated.
    """
        STATE_UNSPECIFIED = 0
        COMPLETED = 1
        SUPERSEDED = 2
        TERMINATED = 3
    duration = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)