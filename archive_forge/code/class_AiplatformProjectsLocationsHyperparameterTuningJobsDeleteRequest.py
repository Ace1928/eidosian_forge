from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsHyperparameterTuningJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsHyperparameterTuningJobsDeleteRequest
  object.

  Fields:
    name: Required. The name of the HyperparameterTuningJob resource to be
      deleted. Format: `projects/{project}/locations/{location}/hyperparameter
      TuningJobs/{hyperparameter_tuning_job}`
  """
    name = _messages.StringField(1, required=True)