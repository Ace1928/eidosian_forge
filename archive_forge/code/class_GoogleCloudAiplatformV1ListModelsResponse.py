from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListModelsResponse(_messages.Message):
    """Response message for ModelService.ListModels

  Fields:
    models: List of Models in the requested page.
    nextPageToken: A token to retrieve next page of results. Pass to
      ListModelsRequest.page_token to obtain that page.
  """
    models = _messages.MessageField('GoogleCloudAiplatformV1Model', 1, repeated=True)
    nextPageToken = _messages.StringField(2)