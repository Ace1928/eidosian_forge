from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListFeatureGroupsResponse(_messages.Message):
    """Response message for FeatureRegistryService.ListFeatureGroups.

  Fields:
    featureGroups: The FeatureGroups matching the request.
    nextPageToken: A token, which can be sent as
      ListFeatureGroupsRequest.page_token to retrieve the next page. If this
      field is omitted, there are no subsequent pages.
  """
    featureGroups = _messages.MessageField('GoogleCloudAiplatformV1FeatureGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)