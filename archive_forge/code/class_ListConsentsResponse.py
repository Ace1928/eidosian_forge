from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConsentsResponse(_messages.Message):
    """A ListConsentsResponse object.

  Fields:
    consents: The returned Consents. The maximum number of Consents returned
      is determined by the value of page_size in the ListConsentsRequest.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    consents = _messages.MessageField('Consent', 1, repeated=True)
    nextPageToken = _messages.StringField(2)