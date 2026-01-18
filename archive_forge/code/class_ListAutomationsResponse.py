from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAutomationsResponse(_messages.Message):
    """The response object from `ListAutomations`.

  Fields:
    automations: The `Automation` objects.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Locations that could not be reached.
  """
    automations = _messages.MessageField('Automation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)