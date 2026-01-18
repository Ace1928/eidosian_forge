from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesDnsRecordSetsListRequest(_messages.Message):
    """A ServicenetworkingServicesDnsRecordSetsListRequest object.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is the project
      number, as in '12345' {network} is the network name.
    parent: Required. The service that is managing peering connectivity for a
      service producer's organization. For Google services that support this
      functionality, this value is
      `services/servicenetworking.googleapis.com`.
    zone: Required. The name of the private DNS zone in the shared producer
      host project from which the record set will be removed.
  """
    consumerNetwork = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)
    zone = _messages.StringField(3)