from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListExtensionsResponse(_messages.Message):
    """Response message for ExtensionRegistryService.ListExtensions

  Fields:
    extensions: List of Extension in the requested page.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListExtensionsRequest.page_token to obtain that page.
  """
    extensions = _messages.MessageField('GoogleCloudAiplatformV1beta1Extension', 1, repeated=True)
    nextPageToken = _messages.StringField(2)