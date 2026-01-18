from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketMirroringForwardingRuleInfo(_messages.Message):
    """A PacketMirroringForwardingRuleInfo object.

  Fields:
    canonicalUrl: [Output Only] Unique identifier for the forwarding rule;
      defined by the server.
    url: Resource URL to the forwarding rule representing the ILB configured
      as destination of the mirrored traffic.
  """
    canonicalUrl = _messages.StringField(1)
    url = _messages.StringField(2)