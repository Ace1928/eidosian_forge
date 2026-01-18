from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListHyperparameterTuningJobsResponse(_messages.Message):
    """Response message for JobService.ListHyperparameterTuningJobs

  Fields:
    hyperparameterTuningJobs: List of HyperparameterTuningJobs in the
      requested page. HyperparameterTuningJob.trials of the jobs will be not
      be returned.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListHyperparameterTuningJobsRequest.page_token to obtain that page.
  """
    hyperparameterTuningJobs = _messages.MessageField('GoogleCloudAiplatformV1HyperparameterTuningJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)