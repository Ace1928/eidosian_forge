from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTrainingPipelinesCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsTrainingPipelinesCancelRequest object.

  Fields:
    googleCloudAiplatformV1CancelTrainingPipelineRequest: A
      GoogleCloudAiplatformV1CancelTrainingPipelineRequest resource to be
      passed as the request body.
    name: Required. The name of the TrainingPipeline to cancel. Format: `proje
      cts/{project}/locations/{location}/trainingPipelines/{training_pipeline}
      `
  """
    googleCloudAiplatformV1CancelTrainingPipelineRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelTrainingPipelineRequest', 1)
    name = _messages.StringField(2, required=True)