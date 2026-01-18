from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsAddContextArtifactsAndExecutionsRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsAddContextArtifactsAn
  dExecutionsRequest object.

  Fields:
    context: Required. The resource name of the Context that the Artifacts and
      Executions belong to. Format: `projects/{project}/locations/{location}/m
      etadataStores/{metadatastore}/contexts/{context}`
    googleCloudAiplatformV1AddContextArtifactsAndExecutionsRequest: A
      GoogleCloudAiplatformV1AddContextArtifactsAndExecutionsRequest resource
      to be passed as the request body.
  """
    context = _messages.StringField(1, required=True)
    googleCloudAiplatformV1AddContextArtifactsAndExecutionsRequest = _messages.MessageField('GoogleCloudAiplatformV1AddContextArtifactsAndExecutionsRequest', 2)