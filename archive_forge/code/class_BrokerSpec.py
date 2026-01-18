from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BrokerSpec(_messages.Message):
    """A BrokerSpec object.

  Fields:
    config: Config is a KReference to the configuration that specifies
      configuration options for this Broker. For example, this could be a
      pointer to a ConfigMap.
    delivery: Delivery is the delivery specification for Events within the
      Broker mesh. This includes things like retries, DLQ, etc.
  """
    config = _messages.MessageField('KReference', 1)
    delivery = _messages.MessageField('DeliverySpec', 2)