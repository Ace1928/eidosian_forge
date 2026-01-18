from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMigratableResourcesSearchRequest(_messages.Message):
    """A AiplatformProjectsLocationsMigratableResourcesSearchRequest object.

  Fields:
    googleCloudAiplatformV1SearchMigratableResourcesRequest: A
      GoogleCloudAiplatformV1SearchMigratableResourcesRequest resource to be
      passed as the request body.
    parent: Required. The location that the migratable resources should be
      searched from. It's the Vertex AI location that the resources can be
      migrated to, not the resources' original location. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1SearchMigratableResourcesRequest = _messages.MessageField('GoogleCloudAiplatformV1SearchMigratableResourcesRequest', 1)
    parent = _messages.StringField(2, required=True)