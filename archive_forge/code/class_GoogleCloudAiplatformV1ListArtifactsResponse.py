from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListArtifactsResponse(_messages.Message):
    """Response message for MetadataService.ListArtifacts.

  Fields:
    artifacts: The Artifacts retrieved from the MetadataStore.
    nextPageToken: A token, which can be sent as
      ListArtifactsRequest.page_token to retrieve the next page. If this field
      is not populated, there are no subsequent pages.
  """
    artifacts = _messages.MessageField('GoogleCloudAiplatformV1Artifact', 1, repeated=True)
    nextPageToken = _messages.StringField(2)