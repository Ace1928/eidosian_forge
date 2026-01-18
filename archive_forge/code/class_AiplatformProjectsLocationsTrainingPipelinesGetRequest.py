from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTrainingPipelinesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsTrainingPipelinesGetRequest object.

  Fields:
    name: Required. The name of the TrainingPipeline resource. Format: `projec
      ts/{project}/locations/{location}/trainingPipelines/{training_pipeline}`
  """
    name = _messages.StringField(1, required=True)