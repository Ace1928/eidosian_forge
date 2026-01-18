from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZonePeeringConfig(_messages.Message):
    """A ManagedZonePeeringConfig object.

  Fields:
    kind: A string attribute.
    targetNetwork: The network with which to peer.
  """
    kind = _messages.StringField(1, default='dns#managedZonePeeringConfig')
    targetNetwork = _messages.MessageField('ManagedZonePeeringConfigTargetNetwork', 2)