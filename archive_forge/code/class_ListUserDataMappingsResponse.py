from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUserDataMappingsResponse(_messages.Message):
    """A ListUserDataMappingsResponse object.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    userDataMappings: The returned User data mappings. The maximum number of
      User data mappings returned is determined by the value of page_size in
      the ListUserDataMappingsRequest.
  """
    nextPageToken = _messages.StringField(1)
    userDataMappings = _messages.MessageField('UserDataMapping', 2, repeated=True)