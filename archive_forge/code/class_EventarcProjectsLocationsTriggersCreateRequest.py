from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsTriggersCreateRequest(_messages.Message):
    """A EventarcProjectsLocationsTriggersCreateRequest object.

  Fields:
    parent: Required. The parent collection in which to add this trigger.
    trigger: A Trigger resource to be passed as the request body.
    triggerId: Required. The user-provided ID to be assigned to the trigger.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not post it.
  """
    parent = _messages.StringField(1, required=True)
    trigger = _messages.MessageField('Trigger', 2)
    triggerId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)