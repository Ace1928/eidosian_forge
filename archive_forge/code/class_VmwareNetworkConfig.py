from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareNetworkConfig(_messages.Message):
    """Specifies network config for the VMware user cluster.

  Fields:
    controlPlaneV2Config: Configuration for control plane V2 mode.
    dhcpIpConfig: Configuration settings for a DHCP IP configuration.
    hostConfig: Represents common network settings irrespective of the host's
      IP address.
    podAddressCidrBlocks: Required. All pods in the cluster are assigned an
      RFC1918 IPv4 address from these ranges. Only a single range is
      supported. This field cannot be changed after creation.
    serviceAddressCidrBlocks: Required. All services in the cluster are
      assigned an RFC1918 IPv4 address from these ranges. Only a single range
      is supported. This field cannot be changed after creation.
    staticIpConfig: Configuration settings for a static IP configuration.
    vcenterNetwork: vcenter_network specifies vCenter network name. Inherited
      from the admin cluster.
  """
    controlPlaneV2Config = _messages.MessageField('VmwareControlPlaneV2Config', 1)
    dhcpIpConfig = _messages.MessageField('VmwareDhcpIpConfig', 2)
    hostConfig = _messages.MessageField('VmwareHostConfig', 3)
    podAddressCidrBlocks = _messages.StringField(4, repeated=True)
    serviceAddressCidrBlocks = _messages.StringField(5, repeated=True)
    staticIpConfig = _messages.MessageField('VmwareStaticIpConfig', 6)
    vcenterNetwork = _messages.StringField(7)