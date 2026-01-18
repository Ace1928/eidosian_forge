from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAddressPool(_messages.Message):
    """Represents an IP pool used by the load balancer.

  Fields:
    addresses: Required. The addresses that are part of this pool. Each
      address must be either in the CIDR form (1.2.3.0/24) or range form
      (1.2.3.1-1.2.3.5).
    avoidBuggyIps: If true, avoid using IPs ending in .0 or .255. This avoids
      buggy consumer devices mistakenly dropping IPv4 traffic for those
      special IP addresses.
    manualAssign: If true, prevent IP addresses from being automatically
      assigned.
    pool: Required. The name of the address pool.
  """
    addresses = _messages.StringField(1, repeated=True)
    avoidBuggyIps = _messages.BooleanField(2)
    manualAssign = _messages.BooleanField(3)
    pool = _messages.StringField(4)