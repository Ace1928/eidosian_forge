from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTriggersResponse(_messages.Message):
    """The response message for the `ListTriggers` method.

  Fields:
    nextPageToken: A page token that can be sent to `ListTriggers` to request
      the next page. If this is empty, then there are no more pages.
    triggers: The requested triggers, up to the number specified in
      `page_size`.
    unreachable: Unreachable resources, if any.
  """
    nextPageToken = _messages.StringField(1)
    triggers = _messages.MessageField('Trigger', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)