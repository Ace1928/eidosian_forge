from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeBillingOwnerValueValuesEnum(_messages.Enum):
    """The Compute Billing Owner for this Data Boost App Profile.

    Values:
      COMPUTE_BILLING_OWNER_UNSPECIFIED: Unspecified value.
      HOST_PAYS: The host Cloud Project containing the targeted Bigtable
        Instance / Table pays for compute.
      REQUESTER_PAYS: The requester Cloud Project targeting the Bigtable
        Instance / Table with Data Boost pays for compute.
    """
    COMPUTE_BILLING_OWNER_UNSPECIFIED = 0
    HOST_PAYS = 1
    REQUESTER_PAYS = 2