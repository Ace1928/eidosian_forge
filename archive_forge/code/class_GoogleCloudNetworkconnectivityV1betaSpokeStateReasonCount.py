from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaSpokeStateReasonCount(_messages.Message):
    """The number of spokes in the hub that are inactive for this reason.

  Enums:
    StateReasonCodeValueValuesEnum: Output only. The reason that a spoke is
      inactive.

  Fields:
    count: Output only. The total number of spokes that are inactive for a
      particular reason and associated with a given hub.
    stateReasonCode: Output only. The reason that a spoke is inactive.
  """

    class StateReasonCodeValueValuesEnum(_messages.Enum):
        """Output only. The reason that a spoke is inactive.

    Values:
      CODE_UNSPECIFIED: No information available.
      PENDING_REVIEW: The proposed spoke is pending review.
      REJECTED: The proposed spoke has been rejected by the hub administrator.
      PAUSED: The spoke has been deactivated internally.
      FAILED: Network Connectivity Center encountered errors while accepting
        the spoke.
    """
        CODE_UNSPECIFIED = 0
        PENDING_REVIEW = 1
        REJECTED = 2
        PAUSED = 3
        FAILED = 4
    count = _messages.IntegerField(1)
    stateReasonCode = _messages.EnumField('StateReasonCodeValueValuesEnum', 2)