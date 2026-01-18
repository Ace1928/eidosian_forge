from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsDatasetVersionsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsDatasetVersionsCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1DatasetVersion: A
      GoogleCloudAiplatformV1DatasetVersion resource to be passed as the
      request body.
    parent: Required. The name of the Dataset resource. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`
  """
    googleCloudAiplatformV1DatasetVersion = _messages.MessageField('GoogleCloudAiplatformV1DatasetVersion', 1)
    parent = _messages.StringField(2, required=True)