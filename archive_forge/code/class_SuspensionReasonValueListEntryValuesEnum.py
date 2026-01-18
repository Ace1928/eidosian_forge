from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SuspensionReasonValueListEntryValuesEnum(_messages.Enum):
    """SuspensionReasonValueListEntryValuesEnum enum type.

    Values:
      SQL_SUSPENSION_REASON_UNSPECIFIED: This is an unknown suspension reason.
      BILLING_ISSUE: The instance is suspended due to billing issues (for
        example:, GCP account issue)
      LEGAL_ISSUE: The instance is suspended due to illegal content (for
        example:, child pornography, copyrighted material, etc.).
      OPERATIONAL_ISSUE: The instance is causing operational issues (for
        example:, causing the database to crash).
      KMS_KEY_ISSUE: The KMS key used by the instance is either revoked or
        denied access to
    """
    SQL_SUSPENSION_REASON_UNSPECIFIED = 0
    BILLING_ISSUE = 1
    LEGAL_ISSUE = 2
    OPERATIONAL_ISSUE = 3
    KMS_KEY_ISSUE = 4