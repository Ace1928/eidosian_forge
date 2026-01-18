from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsBatchPredictionJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsBatchPredictionJobsDeleteRequest object.

  Fields:
    name: Required. The name of the BatchPredictionJob resource to be deleted.
      Format: `projects/{project}/locations/{location}/batchPredictionJobs/{ba
      tch_prediction_job}`
  """
    name = _messages.StringField(1, required=True)