from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSecurityZerotrustApplinkAppConnectorProtoGateway(_messages.Message):
    """Gateway represents a GCE VM Instance endpoint for use by IAP TCP.

  Fields:
    interface: interface specifies the network interface of the gateway to
      connect to.
    name: name is the name of an instance running a gateway. It is the unique
      ID for a gateway. All gateways under the same connection have the same
      prefix. It is derived from the gateway URL. For example,
      name=${instance} assuming a gateway URL. https://www.googleapis.com/comp
      ute/${version}/projects/${project}/zones/${zone}/instances/${instance}
    port: port specifies the port of the gateway for tunnel connections from
      the connectors.
    project: project is the tenant project the gateway belongs to. Different
      from the project in the connection, it is a BeyondCorpAPI internally
      created project to manage all the gateways. It is sharing the same
      network with the consumer project user owned. It is derived from the
      gateway URL. For example, project=${project} assuming a gateway URL. htt
      ps://www.googleapis.com/compute/${version}/projects/${project}/zones/${z
      one}/instances/${instance}
    selfLink: self_link is the gateway URL in the form https://www.googleapis.
      com/compute/${version}/projects/${project}/zones/${zone}/instances/${ins
      tance}
    zone: zone represents the zone the instance belongs. It is derived from
      the gateway URL. For example, zone=${zone} assuming a gateway URL. https
      ://www.googleapis.com/compute/${version}/projects/${project}/zones/${zon
      e}/instances/${instance}
  """
    interface = _messages.StringField(1)
    name = _messages.StringField(2)
    port = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    project = _messages.StringField(4)
    selfLink = _messages.StringField(5)
    zone = _messages.StringField(6)