from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataBoostIsolationReadOnly(_messages.Message):
    """Data Boost is a serverless compute capability that lets you run high-
  throughput read jobs on your Bigtable data, without impacting the
  performance of the clusters that handle your application traffic. Currently,
  Data Boost exclusively supports read-only use-cases with single-cluster
  routing. Data Boost reads are only guaranteed to see the results of writes
  that were written at least 30 minutes ago. This means newly written values
  may not become visible for up to 30m, and also means that old values may
  remain visible for up to 30m after being deleted or overwritten. To mitigate
  the staleness of the data, users may either wait 30m, or use
  CheckConsistency.

  Enums:
    ComputeBillingOwnerValueValuesEnum: The Compute Billing Owner for this
      Data Boost App Profile.

  Fields:
    computeBillingOwner: The Compute Billing Owner for this Data Boost App
      Profile.
  """

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
    computeBillingOwner = _messages.EnumField('ComputeBillingOwnerValueValuesEnum', 1)