from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProxyConnectionInfo(_messages.Message):
    """For display only. Metadata associated with ProxyConnection.

  Fields:
    networkUri: URI of the network where connection is proxied.
    newDestinationIp: Destination IP address of a new connection.
    newDestinationPort: Destination port of a new connection. Only valid when
      protocol is TCP or UDP.
    newSourceIp: Source IP address of a new connection.
    newSourcePort: Source port of a new connection. Only valid when protocol
      is TCP or UDP.
    oldDestinationIp: Destination IP address of an original connection
    oldDestinationPort: Destination port of an original connection. Only valid
      when protocol is TCP or UDP.
    oldSourceIp: Source IP address of an original connection.
    oldSourcePort: Source port of an original connection. Only valid when
      protocol is TCP or UDP.
    protocol: IP protocol in string format, for example: "TCP", "UDP", "ICMP".
    subnetUri: Uri of proxy subnet.
  """
    networkUri = _messages.StringField(1)
    newDestinationIp = _messages.StringField(2)
    newDestinationPort = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    newSourceIp = _messages.StringField(4)
    newSourcePort = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    oldDestinationIp = _messages.StringField(6)
    oldDestinationPort = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    oldSourceIp = _messages.StringField(8)
    oldSourcePort = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    protocol = _messages.StringField(10)
    subnetUri = _messages.StringField(11)