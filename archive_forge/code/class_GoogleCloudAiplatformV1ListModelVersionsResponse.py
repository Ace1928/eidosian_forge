from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListModelVersionsResponse(_messages.Message):
    """Response message for ModelService.ListModelVersions

  Fields:
    models: List of Model versions in the requested page. In the returned
      Model name field, version ID instead of regvision tag will be included.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListModelVersionsRequest.page_token to obtain that page.
  """
    models = _messages.MessageField('GoogleCloudAiplatformV1Model', 1, repeated=True)
    nextPageToken = _messages.StringField(2)