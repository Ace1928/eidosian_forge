from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateDnsRecordSetRequest(_messages.Message):
    """Request to update a record set from a private managed DNS zone in the
  shared producer host project. The name, type, ttl, and data values of the
  existing record set must all exactly match an existing record set in the
  specified zone.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is the project
      number, as in '12345' {network} is the network name.
    existingDnsRecordSet: Required. The existing DNS record set to update.
    newDnsRecordSet: Required. The new values that the DNS record set should
      be updated to hold.
    zone: Required. The name of the private DNS zone in the shared producer
      host project from which the record set will be removed.
  """
    consumerNetwork = _messages.StringField(1)
    existingDnsRecordSet = _messages.MessageField('DnsRecordSet', 2)
    newDnsRecordSet = _messages.MessageField('DnsRecordSet', 3)
    zone = _messages.StringField(4)