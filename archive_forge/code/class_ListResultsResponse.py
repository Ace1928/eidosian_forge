from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListResultsResponse(_messages.Message):
    """Message for response to listing Results.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    results: The list of Results.
  """
    nextPageToken = _messages.StringField(1)
    results = _messages.MessageField('Result', 2, repeated=True)