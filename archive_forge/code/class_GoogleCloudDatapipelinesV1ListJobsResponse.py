from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1ListJobsResponse(_messages.Message):
    """Response message for ListJobs

  Fields:
    jobs: Results that were accessible to the caller. Results are always in
      descending order of job creation date.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    jobs = _messages.MessageField('GoogleCloudDatapipelinesV1Job', 1, repeated=True)
    nextPageToken = _messages.StringField(2)