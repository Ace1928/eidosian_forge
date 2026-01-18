from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListBatchPredictionJobsResponse(_messages.Message):
    """Response message for JobService.ListBatchPredictionJobs

  Fields:
    batchPredictionJobs: List of BatchPredictionJobs in the requested page.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListBatchPredictionJobsRequest.page_token to obtain that page.
  """
    batchPredictionJobs = _messages.MessageField('GoogleCloudAiplatformV1beta1BatchPredictionJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)