from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareLoadBalancerConfig(_messages.Message):
    """Specifies the locad balancer config for the VMware user cluster.

  Fields:
    f5Config: Configuration for F5 Big IP typed load balancers.
    manualLbConfig: Manually configured load balancers.
    metalLbConfig: Configuration for MetalLB typed load balancers.
    seesawConfig: Output only. Configuration for Seesaw typed load balancers.
    vipConfig: The VIPs used by the load balancer.
  """
    f5Config = _messages.MessageField('VmwareF5BigIpConfig', 1)
    manualLbConfig = _messages.MessageField('VmwareManualLbConfig', 2)
    metalLbConfig = _messages.MessageField('VmwareMetalLbConfig', 3)
    seesawConfig = _messages.MessageField('VmwareSeesawConfig', 4)
    vipConfig = _messages.MessageField('VmwareVipConfig', 5)