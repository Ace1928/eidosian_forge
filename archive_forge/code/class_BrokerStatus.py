from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BrokerStatus(_messages.Message):
    """A BrokerStatus object.

  Fields:
    address: Broker is Addressable. It exposes the endpoint as an URI to get
      events delivered into the Broker mesh.
    conditions: Conditions the latest available observations of a resource's
      current state.
    observedGeneration: ObservedGeneration is the 'Generation' of the Broker
      that was last processed by the controller.
  """
    address = _messages.MessageField('Addressable', 1)
    conditions = _messages.MessageField('Condition', 2, repeated=True)
    observedGeneration = _messages.IntegerField(3, variant=_messages.Variant.INT32)