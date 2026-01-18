from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPlatformsResponse(_messages.Message):
    """Response message for PlatformPolicyManagementService.ListPlatforms.

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass this
      value in the ListPlatformsRequest.page_token field in the subsequent
      call to the `ListPlatforms` method to retrieve the next page of results.
    platforms: The list of platforms supported by Binary Authorization.
  """
    nextPageToken = _messages.StringField(1)
    platforms = _messages.MessageField('Platform', 2, repeated=True)