from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsPurgeRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsPurgeRequest object.

  Fields:
    googleCloudAiplatformV1PurgeContextsRequest: A
      GoogleCloudAiplatformV1PurgeContextsRequest resource to be passed as the
      request body.
    parent: Required. The metadata store to purge Contexts from. Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    googleCloudAiplatformV1PurgeContextsRequest = _messages.MessageField('GoogleCloudAiplatformV1PurgeContextsRequest', 1)
    parent = _messages.StringField(2, required=True)