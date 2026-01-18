from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceState(_messages.Message):
    """Information about the state of a service with respect to a consumer
  resource.

  Fields:
    name: Output only. The resource name of the service state.
    service: Output only. The service referenced by this state.
    state: Output only. The state of this service with respect to the consumer
      parent.
  """
    name = _messages.StringField(1)
    service = _messages.MessageField('Service', 2)
    state = _messages.MessageField('State', 3)