from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateIpv6GoogleAccessValueValuesEnum(_messages.Enum):
    """Optional. The private IPv6 google access type for the VM. If not
    specified, use INHERIT_FROM_SUBNETWORK as default.

    Values:
      INSTANCE_PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED: Default value. This
        value is unused.
      INHERIT_FROM_SUBNETWORK: Each network interface inherits
        PrivateIpv6GoogleAccess from its subnetwork.
      ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE: Outbound private IPv6 access from
        VMs in this subnet to Google services. If specified, the subnetwork
        who is attached to the instance's default network interface will be
        assigned an internal IPv6 prefix if it doesn't have before.
      ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE: Bidirectional private IPv6 access
        to/from Google services. If specified, the subnetwork who is attached
        to the instance's default network interface will be assigned an
        internal IPv6 prefix if it doesn't have before.
    """
    INSTANCE_PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED = 0
    INHERIT_FROM_SUBNETWORK = 1
    ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE = 2
    ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE = 3