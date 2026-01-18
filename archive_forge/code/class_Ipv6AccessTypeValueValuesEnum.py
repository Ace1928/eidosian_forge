from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Ipv6AccessTypeValueValuesEnum(_messages.Enum):
    """Optional. [Output Only] One of EXTERNAL, INTERNAL to indicate whether
    the IP can be accessed from the Internet. This field is always inherited
    from its subnetwork.

    Values:
      UNSPECIFIED_IPV6_ACCESS_TYPE: IPv6 access type not set. Means this
        network interface hasn't been turned on IPv6 yet.
      INTERNAL: This network interface can have internal IPv6.
      EXTERNAL: This network interface can have external IPv6.
    """
    UNSPECIFIED_IPV6_ACCESS_TYPE = 0
    INTERNAL = 1
    EXTERNAL = 2