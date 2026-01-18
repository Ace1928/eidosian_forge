from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NatIpAllocateOptionValueValuesEnum(_messages.Enum):
    """Specify the NatIpAllocateOption, which can take one of the following
    values: - MANUAL_ONLY: Uses only Nat IP addresses provided by customers.
    When there are not enough specified Nat IPs, the Nat service fails for new
    VMs. - AUTO_ONLY: Nat IPs are allocated by Google Cloud Platform;
    customers can't specify any Nat IPs. When choosing AUTO_ONLY, then nat_ip
    should be empty.

    Values:
      AUTO_ONLY: Nat IPs are allocated by GCP; customers can not specify any
        Nat IPs.
      MANUAL_ONLY: Only use Nat IPs provided by customers. When specified Nat
        IPs are not enough then the Nat service fails for new VMs.
    """
    AUTO_ONLY = 0
    MANUAL_ONLY = 1