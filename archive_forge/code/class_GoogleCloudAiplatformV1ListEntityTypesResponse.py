from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListEntityTypesResponse(_messages.Message):
    """Response message for FeaturestoreService.ListEntityTypes.

  Fields:
    entityTypes: The EntityTypes matching the request.
    nextPageToken: A token, which can be sent as
      ListEntityTypesRequest.page_token to retrieve the next page. If this
      field is omitted, there are no subsequent pages.
  """
    entityTypes = _messages.MessageField('GoogleCloudAiplatformV1EntityType', 1, repeated=True)
    nextPageToken = _messages.StringField(2)