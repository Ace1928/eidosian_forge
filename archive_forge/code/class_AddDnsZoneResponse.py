from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddDnsZoneResponse(_messages.Message):
    """Represents managed DNS zones created in the shared producer host and
  consumer projects.

  Fields:
    consumerPeeringZone: The DNS peering zone created in the consumer project.
    producerPrivateZone: The private DNS zone created in the shared producer
      host project.
  """
    consumerPeeringZone = _messages.MessageField('DnsZone', 1)
    producerPrivateZone = _messages.MessageField('DnsZone', 2)