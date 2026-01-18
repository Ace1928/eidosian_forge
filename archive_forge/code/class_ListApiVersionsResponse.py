from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListApiVersionsResponse(_messages.Message):
    """Response message for ListApiVersions.

  Fields:
    apiVersions: The versions from the specified publisher.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    apiVersions = _messages.MessageField('ApiVersion', 1, repeated=True)
    nextPageToken = _messages.StringField(2)