from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalLoadBalancerPool(_messages.Message):
    """External load balancer pool with custom config such as name, manual/auto
  assign, non-overlapping ipv4 and optional ipv6 address range.

  Fields:
    addressPool: Optional. Name of the external load balancer pool.
    avoidBuggyIps: Optional. If true, the pool omits IP addresses ending in .0
      and .255. Some network hardware drops traffic to these special
      addresses. Its default value is false.
    ipv4Range: Required. Non-overlapping IPv4 address range of the external
      load balancer pool.
    ipv6Range: Optional. Non-overlapping IPv6 address range of the external
      load balancer pool.
    manualAssign: Optional. If true, addresses in this pool are not
      automatically assigned to Kubernetes Services. If true, an IP address in
      this pool is used only when it is specified explicitly by a service. Its
      default value is false.
  """
    addressPool = _messages.StringField(1)
    avoidBuggyIps = _messages.BooleanField(2)
    ipv4Range = _messages.StringField(3, repeated=True)
    ipv6Range = _messages.StringField(4, repeated=True)
    manualAssign = _messages.BooleanField(5)