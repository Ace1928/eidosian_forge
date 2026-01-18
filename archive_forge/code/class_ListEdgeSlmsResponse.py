from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEdgeSlmsResponse(_messages.Message):
    """Message for response to listing EdgeSlms.

  Fields:
    edgeSlms: The list of EdgeSlm
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    edgeSlms = _messages.MessageField('EdgeSlm', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)