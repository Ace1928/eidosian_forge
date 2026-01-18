from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ContainerRegistryDestination(_messages.Message):
    """The Container Registry location for the container image.

  Fields:
    outputUri: Required. Container Registry URI of a container image. Only
      Google Container Registry and Artifact Registry are supported now.
      Accepted forms: * Google Container Registry path. For example:
      `gcr.io/projectId/imageName:tag`. * Artifact Registry path. For example:
      `us-central1-docker.pkg.dev/projectId/repoName/imageName:tag`. If a tag
      is not specified, "latest" will be used as the default tag.
  """
    outputUri = _messages.StringField(1)