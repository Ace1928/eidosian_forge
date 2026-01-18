from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventingRuntimeData(_messages.Message):
    """Eventing runtime data has the details related to eventing managed by the
  system.

  Fields:
    eventsListenerEndpoint: Output only. Events listener endpoint. The value
      will populated after provisioning the events listener.
    status: Output only. Current status of eventing.
  """
    eventsListenerEndpoint = _messages.StringField(1)
    status = _messages.MessageField('EventingStatus', 2)