from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsHyperparameterTuningJobsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsHyperparameterTuningJobsGetRequest object.

  Fields:
    name: Required. The name of the HyperparameterTuningJob resource. Format:
      `projects/{project}/locations/{location}/hyperparameterTuningJobs/{hyper
      parameter_tuning_job}`
  """
    name = _messages.StringField(1, required=True)