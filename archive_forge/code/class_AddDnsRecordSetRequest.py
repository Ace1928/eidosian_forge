from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddDnsRecordSetRequest(_messages.Message):
    """Request to add a record set to a private managed DNS zone in the shared
  producer host project.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is the project
      number, as in '12345' {network} is the network name.
    dnsRecordSet: Required. The DNS record set to add.
    zone: Required. The name of the private DNS zone in the shared producer
      host project to which the record set will be added.
  """
    consumerNetwork = _messages.StringField(1)
    dnsRecordSet = _messages.MessageField('DnsRecordSet', 2)
    zone = _messages.StringField(3)