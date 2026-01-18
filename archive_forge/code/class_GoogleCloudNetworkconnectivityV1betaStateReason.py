from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaStateReason(_messages.Message):
    """The reason a spoke is inactive.

  Enums:
    CodeValueValuesEnum: The code associated with this reason.

  Fields:
    code: The code associated with this reason.
    message: Human-readable details about this reason.
    userDetails: Additional information provided by the user in the
      RejectSpoke call.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """The code associated with this reason.

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
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    message = _messages.StringField(2)
    userDetails = _messages.StringField(3)