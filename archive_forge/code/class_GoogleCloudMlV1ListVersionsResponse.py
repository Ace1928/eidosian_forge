from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListVersionsResponse(_messages.Message):
    """Response message for the ListVersions method.

  Fields:
    nextPageToken: Optional. Pass this token as the `page_token` field of the
      request for a subsequent call.
    versions: The list of versions.
  """
    nextPageToken = _messages.StringField(1)
    versions = _messages.MessageField('GoogleCloudMlV1Version', 2, repeated=True)