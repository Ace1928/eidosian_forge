from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPersistentResourcesRebootRequest(_messages.Message):
    """A AiplatformProjectsLocationsPersistentResourcesRebootRequest object.

  Fields:
    googleCloudAiplatformV1RebootPersistentResourceRequest: A
      GoogleCloudAiplatformV1RebootPersistentResourceRequest resource to be
      passed as the request body.
    name: Required. The name of the PersistentResource resource. Format: `proj
      ects/{project_id_or_number}/locations/{location_id}/persistentResources/
      {persistent_resource_id}`
  """
    googleCloudAiplatformV1RebootPersistentResourceRequest = _messages.MessageField('GoogleCloudAiplatformV1RebootPersistentResourceRequest', 1)
    name = _messages.StringField(2, required=True)