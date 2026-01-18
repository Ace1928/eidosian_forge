from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsHyperparameterTuningJobsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsHyperparameterTuningJobsCreateRequest
  object.

  Fields:
    googleCloudAiplatformV1HyperparameterTuningJob: A
      GoogleCloudAiplatformV1HyperparameterTuningJob resource to be passed as
      the request body.
    parent: Required. The resource name of the Location to create the
      HyperparameterTuningJob in. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1HyperparameterTuningJob = _messages.MessageField('GoogleCloudAiplatformV1HyperparameterTuningJob', 1)
    parent = _messages.StringField(2, required=True)