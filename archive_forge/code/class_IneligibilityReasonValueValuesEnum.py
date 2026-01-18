from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IneligibilityReasonValueValuesEnum(_messages.Enum):
    """The reason of why it is ineligible to request increased value of the
    quota. If the is_eligible field is true, it defaults to
    INELIGIBILITY_REASON_UNSPECIFIED.

    Values:
      INELIGIBILITY_REASON_UNSPECIFIED: Default value when is_eligible is
        true.
      NO_VALID_BILLING_ACCOUNT: The container is not linked with a valid
        billing account.
      OTHER: Other reasons.
    """
    INELIGIBILITY_REASON_UNSPECIFIED = 0
    NO_VALID_BILLING_ACCOUNT = 1
    OTHER = 2