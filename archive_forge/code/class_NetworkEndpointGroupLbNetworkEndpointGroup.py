from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupLbNetworkEndpointGroup(_messages.Message):
    """Load balancing specific fields for network endpoint group.

  Fields:
    defaultPort: The default port used if the port number is not specified in
      the network endpoint. If the network endpoint type is either GCE_VM_IP,
      SERVERLESS or PRIVATE_SERVICE_CONNECT, this field must not be specified.
      [Deprecated] This field is deprecated.
    network: The URL of the network to which all network endpoints in the NEG
      belong. Uses default project network if unspecified. [Deprecated] This
      field is deprecated.
    subnetwork: Optional URL of the subnetwork to which all network endpoints
      in the NEG belong. [Deprecated] This field is deprecated.
    zone: [Output Only] The URL of the zone where the network endpoint group
      is located. [Deprecated] This field is deprecated.
  """
    defaultPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    network = _messages.StringField(2)
    subnetwork = _messages.StringField(3)
    zone = _messages.StringField(4)