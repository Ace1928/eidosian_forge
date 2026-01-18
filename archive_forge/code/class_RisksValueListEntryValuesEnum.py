from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RisksValueListEntryValuesEnum(_messages.Enum):
    """RisksValueListEntryValuesEnum enum type.

    Values:
      RISK_TYPE_UNSPECIFIED: Default unspecified risk. Don't use directly.
      SERVICE_DISRUPTION: Potential service downtime.
      DATA_LOSS: Potential data loss.
      ACCESS_DENY: Potential access denial. The service is still up but some
        or all clients can't access it.
    """
    RISK_TYPE_UNSPECIFIED = 0
    SERVICE_DISRUPTION = 1
    DATA_LOSS = 2
    ACCESS_DENY = 3