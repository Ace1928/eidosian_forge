from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFederationsResponse(_messages.Message):
    """Response message for ListFederations

  Fields:
    federations: The services in the specified location.
    nextPageToken: A token that can be sent as page_token to retrieve the next
      page. If this field is omitted, there are no subsequent pages.
    unreachable: Locations that could not be reached.
  """
    federations = _messages.MessageField('Federation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)