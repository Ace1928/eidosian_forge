from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsBatchPredictionJobsCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsBatchPredictionJobsCancelRequest object.

  Fields:
    googleCloudAiplatformV1CancelBatchPredictionJobRequest: A
      GoogleCloudAiplatformV1CancelBatchPredictionJobRequest resource to be
      passed as the request body.
    name: Required. The name of the BatchPredictionJob to cancel. Format: `pro
      jects/{project}/locations/{location}/batchPredictionJobs/{batch_predicti
      on_job}`
  """
    googleCloudAiplatformV1CancelBatchPredictionJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelBatchPredictionJobRequest', 1)
    name = _messages.StringField(2, required=True)