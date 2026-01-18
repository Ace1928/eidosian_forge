from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListFeaturestoresResponse(_messages.Message):
    """Response message for FeaturestoreService.ListFeaturestores.

  Fields:
    featurestores: The Featurestores matching the request.
    nextPageToken: A token, which can be sent as
      ListFeaturestoresRequest.page_token to retrieve the next page. If this
      field is omitted, there are no subsequent pages.
  """
    featurestores = _messages.MessageField('GoogleCloudAiplatformV1Featurestore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)