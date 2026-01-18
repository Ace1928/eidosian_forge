from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTrainingPipelinesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsTrainingPipelinesDeleteRequest object.

  Fields:
    name: Required. The name of the TrainingPipeline resource to be deleted.
      Format: `projects/{project}/locations/{location}/trainingPipelines/{trai
      ning_pipeline}`
  """
    name = _messages.StringField(1, required=True)