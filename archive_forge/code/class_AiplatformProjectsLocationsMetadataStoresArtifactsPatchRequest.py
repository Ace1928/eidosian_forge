from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresArtifactsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresArtifactsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the Artifact is not found, a new
      Artifact is created.
    googleCloudAiplatformV1Artifact: A GoogleCloudAiplatformV1Artifact
      resource to be passed as the request body.
    name: Output only. The resource name of the Artifact.
    updateMask: Optional. A FieldMask indicating which fields should be
      updated.
  """
    allowMissing = _messages.BooleanField(1)
    googleCloudAiplatformV1Artifact = _messages.MessageField('GoogleCloudAiplatformV1Artifact', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)