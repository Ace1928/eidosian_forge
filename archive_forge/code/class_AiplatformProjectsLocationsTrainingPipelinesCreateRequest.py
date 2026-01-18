from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTrainingPipelinesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsTrainingPipelinesCreateRequest object.

  Fields:
    googleCloudAiplatformV1TrainingPipeline: A
      GoogleCloudAiplatformV1TrainingPipeline resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to create the
      TrainingPipeline in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1TrainingPipeline = _messages.MessageField('GoogleCloudAiplatformV1TrainingPipeline', 1)
    parent = _messages.StringField(2, required=True)