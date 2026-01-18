from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneForwardingConfig(_messages.Message):
    """A ManagedZoneForwardingConfig object.

  Fields:
    kind: A string attribute.
    targetNameServers: List of target name servers to forward to. Cloud DNS
      selects the best available name server if more than one target is given.
  """
    kind = _messages.StringField(1, default='dns#managedZoneForwardingConfig')
    targetNameServers = _messages.MessageField('ManagedZoneForwardingConfigNameServerTarget', 2, repeated=True)