from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DesiredPrivateIpv6GoogleAccessValueValuesEnum(_messages.Enum):
    """The desired state of IPv6 connectivity to Google Services.

    Values:
      PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED: Default value. Same as DISABLED
      PRIVATE_IPV6_GOOGLE_ACCESS_DISABLED: No private access to or from Google
        Services
      PRIVATE_IPV6_GOOGLE_ACCESS_TO_GOOGLE: Enables private IPv6 access to
        Google Services from GKE
      PRIVATE_IPV6_GOOGLE_ACCESS_BIDIRECTIONAL: Enables private IPv6 access to
        and from Google Services
    """
    PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED = 0
    PRIVATE_IPV6_GOOGLE_ACCESS_DISABLED = 1
    PRIVATE_IPV6_GOOGLE_ACCESS_TO_GOOGLE = 2
    PRIVATE_IPV6_GOOGLE_ACCESS_BIDIRECTIONAL = 3