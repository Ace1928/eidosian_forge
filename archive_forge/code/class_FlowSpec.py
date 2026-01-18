from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FlowSpec(_messages.Message):
    """Desired state of a Flow.

  Fields:
    action: Where an action gets delivered to. For example an HTTP endpoint.
    trigger: Contains the event_type, the "resource" path, and the hostname of
      the service hosting the event source. The "resource" includes the event
      source and a path match expression specifying a condition for emitting
      an event.
  """
    action = _messages.MessageField('Action', 1)
    trigger = _messages.MessageField('EventTrigger', 2)