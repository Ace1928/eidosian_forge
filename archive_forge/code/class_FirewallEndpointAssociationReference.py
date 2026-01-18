from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallEndpointAssociationReference(_messages.Message):
    """This is a subset of the FirewallEndpointAssociation message, containing
  fields to be used by the consumer.

  Fields:
    name: Output only. The resource name of the FirewallEndpointAssociation.
      Format: projects/{project}/locations/{location}/firewallEndpointAssociat
      ions/{id}
    network: Output only. The VPC network associated. Format:
      projects/{project}/global/networks/{name}.
  """
    name = _messages.StringField(1)
    network = _messages.StringField(2)