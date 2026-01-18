from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsBatchPredictionJobsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsBatchPredictionJobsGetRequest object.

  Fields:
    name: Required. The name of the BatchPredictionJob resource. Format: `proj
      ects/{project}/locations/{location}/batchPredictionJobs/{batch_predictio
      n_job}`
  """
    name = _messages.StringField(1, required=True)