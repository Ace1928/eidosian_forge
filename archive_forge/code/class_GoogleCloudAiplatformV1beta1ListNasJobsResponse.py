from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListNasJobsResponse(_messages.Message):
    """Response message for JobService.ListNasJobs

  Fields:
    nasJobs: List of NasJobs in the requested page. NasJob.nas_job_output of
      the jobs will not be returned.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListNasJobsRequest.page_token to obtain that page.
  """
    nasJobs = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)