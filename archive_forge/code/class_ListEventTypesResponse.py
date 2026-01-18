from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEventTypesResponse(_messages.Message):
    """Response message for Connectors.ListEventTypes.

  Fields:
    eventTypes: A list of connector versions.
    nextPageToken: Next page token.
  """
    eventTypes = _messages.MessageField('EventType', 1, repeated=True)
    nextPageToken = _messages.StringField(2)