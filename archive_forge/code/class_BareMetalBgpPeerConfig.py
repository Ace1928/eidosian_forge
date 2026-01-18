from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalBgpPeerConfig(_messages.Message):
    """BareMetalBgpPeerConfig represents configuration parameters for a Border
  Gateway Protocol (BGP) peer.

  Fields:
    asn: Required. BGP autonomous system number (ASN) for the network that
      contains the external peer device.
    controlPlaneNodes: The IP address of the control plane node that connects
      to the external peer. If you don't specify any control plane nodes, all
      control plane nodes can connect to the external peer. If you specify one
      or more IP addresses, only the nodes specified participate in peering
      sessions.
    ipAddress: Required. The IP address of the external peer device.
  """
    asn = _messages.IntegerField(1)
    controlPlaneNodes = _messages.StringField(2, repeated=True)
    ipAddress = _messages.StringField(3)