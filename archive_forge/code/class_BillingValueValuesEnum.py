from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingValueValuesEnum(_messages.Enum):
    """Deprecated: This field will be ignored and should not be set.
    Customer's billing structure.

    Values:
      BILLING_UNSPECIFIED: Unknown
      PAY_AS_YOU_GO: User pays a fee per-endpoint.
      ANTHOS_LICENSE: User is paying for Anthos as a whole.
    """
    BILLING_UNSPECIFIED = 0
    PAY_AS_YOU_GO = 1
    ANTHOS_LICENSE = 2