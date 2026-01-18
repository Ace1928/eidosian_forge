from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationRoute(_messages.Message):
    """The setting used to configure ClientGateways. It is adding routes to the
  client's routing table after the connection is established.

  Fields:
    address: Required. The network address of the subnet for which the packet
      is routed to the ClientGateway.
    netmask: Required. The network mask of the subnet for which the packet is
      routed to the ClientGateway.
  """
    address = _messages.StringField(1)
    netmask = _messages.StringField(2)