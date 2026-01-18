from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SuspensionReasonsValueListEntryValuesEnum(_messages.Enum):
    """SuspensionReasonsValueListEntryValuesEnum enum type.

    Values:
      SUSPENSION_REASON_UNSPECIFIED: Not set.
      KMS_KEY_ISSUE: The KMS key used by the instance is either revoked or
        denied access to.
    """
    SUSPENSION_REASON_UNSPECIFIED = 0
    KMS_KEY_ISSUE = 1