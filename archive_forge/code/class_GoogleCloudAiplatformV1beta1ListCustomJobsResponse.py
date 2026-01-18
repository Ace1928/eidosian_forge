from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListCustomJobsResponse(_messages.Message):
    """Response message for JobService.ListCustomJobs

  Fields:
    customJobs: List of CustomJobs in the requested page.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListCustomJobsRequest.page_token to obtain that page.
  """
    customJobs = _messages.MessageField('GoogleCloudAiplatformV1beta1CustomJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)