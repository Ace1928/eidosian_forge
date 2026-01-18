from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSecurityZerotrustApplinkAppConnectorProtoConnectionConfig(_messages.Message):
    """ConnectionConfig represents a Connection Configuration object.

  Fields:
    applicationEndpoint: application_endpoint is the endpoint of the
      application the form of host:port. For example, "localhost:80".
    applicationName: application_name represents the given name of the
      application the connection is connecting with.
    gateway: gateway lists all instances running a gateway in GCP. They all
      connect to a connector on the host.
    name: name is the unique ID for each connection. TODO(b/190732451) returns
      connection name from user-specified name in config. Now, name =
      ${application_name}:${application_endpoint}
    project: project represents the consumer project the connection belongs
      to.
    tunnelsPerGateway: tunnels_per_gateway reflects the number of tunnels
      between a connector and a gateway.
    userPort: user_port specifies the reserved port on gateways for user
      connections.
  """
    applicationEndpoint = _messages.StringField(1)
    applicationName = _messages.StringField(2)
    gateway = _messages.MessageField('CloudSecurityZerotrustApplinkAppConnectorProtoGateway', 3, repeated=True)
    name = _messages.StringField(4)
    project = _messages.StringField(5)
    tunnelsPerGateway = _messages.IntegerField(6, variant=_messages.Variant.UINT32)
    userPort = _messages.IntegerField(7, variant=_messages.Variant.INT32)