from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchGrantsResponse(_messages.Message):
    """Response message for `SearchGrants` method.

  Fields:
    grants: The list of Grants.
    nextPageToken: A token identifying a page of results the server should
      return.
  """
    grants = _messages.MessageField('Grant', 1, repeated=True)
    nextPageToken = _messages.StringField(2)