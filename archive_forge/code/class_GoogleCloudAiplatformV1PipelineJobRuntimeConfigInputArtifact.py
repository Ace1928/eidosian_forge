from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PipelineJobRuntimeConfigInputArtifact(_messages.Message):
    """The type of an input artifact.

  Fields:
    artifactId: Artifact resource id from MLMD. Which is the last portion of
      an artifact resource name: `projects/{project}/locations/{location}/meta
      dataStores/default/artifacts/{artifact_id}`. The artifact must stay
      within the same project, location and default metadatastore as the
      pipeline.
  """
    artifactId = _messages.StringField(1)