from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1AddContextArtifactsAndExecutionsRequest(_messages.Message):
    """Request message for MetadataService.AddContextArtifactsAndExecutions.

  Fields:
    artifacts: The resource names of the Artifacts to attribute to the
      Context. Format: `projects/{project}/locations/{location}/metadataStores
      /{metadatastore}/artifacts/{artifact}`
    executions: The resource names of the Executions to associate with the
      Context. Format: `projects/{project}/locations/{location}/metadataStores
      /{metadatastore}/executions/{execution}`
  """
    artifacts = _messages.StringField(1, repeated=True)
    executions = _messages.StringField(2, repeated=True)