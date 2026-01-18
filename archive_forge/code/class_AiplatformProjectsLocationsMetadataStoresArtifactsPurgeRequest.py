from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresArtifactsPurgeRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresArtifactsPurgeRequest object.

  Fields:
    googleCloudAiplatformV1PurgeArtifactsRequest: A
      GoogleCloudAiplatformV1PurgeArtifactsRequest resource to be passed as
      the request body.
    parent: Required. The metadata store to purge Artifacts from. Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    googleCloudAiplatformV1PurgeArtifactsRequest = _messages.MessageField('GoogleCloudAiplatformV1PurgeArtifactsRequest', 1)
    parent = _messages.StringField(2, required=True)