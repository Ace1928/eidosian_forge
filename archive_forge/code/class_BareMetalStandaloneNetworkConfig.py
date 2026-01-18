from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneNetworkConfig(_messages.Message):
    """Specifies the cluster network configuration.

  Fields:
    advancedNetworking: Enables the use of advanced Anthos networking
      features, such as Bundled Load Balancing with BGP or the egress NAT
      gateway. Setting configuration for advanced networking features will
      automatically set this flag.
    islandModeCidr: Configuration for island mode CIDR. In an island-mode
      network, nodes have unique IP addresses, but pods don't have unique
      addresses across clusters. This doesn't cause problems because pods in
      one cluster never directly communicate with pods in another cluster.
      Instead, there are gateways that mediate between a pod in one cluster
      and a pod in another cluster.
    multipleNetworkInterfacesConfig: Configuration for multiple network
      interfaces.
    srIovConfig: Configuration for SR-IOV.
  """
    advancedNetworking = _messages.BooleanField(1)
    islandModeCidr = _messages.MessageField('BareMetalStandaloneIslandModeCidrConfig', 2)
    multipleNetworkInterfacesConfig = _messages.MessageField('BareMetalStandloneMultipleNetworkInterfacesConfig', 3)
    srIovConfig = _messages.MessageField('BareMetalStandaloneSrIovConfig', 4)