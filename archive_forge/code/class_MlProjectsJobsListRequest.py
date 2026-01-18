from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsJobsListRequest(_messages.Message):
    """A MlProjectsJobsListRequest object.

  Fields:
    filter: Optional. Specifies the subset of jobs to retrieve. You can filter
      on the value of one or more attributes of the job object. For example,
      retrieve jobs with a job identifier that starts with 'census': gcloud
      ai-platform jobs list --filter='jobId:census*' List all failed jobs with
      names that start with 'rnn': gcloud ai-platform jobs list
      --filter='jobId:rnn* AND state:FAILED' For more examples, see the guide
      to monitoring jobs.
    pageSize: Optional. The number of jobs to retrieve per "page" of results.
      If there are more remaining results than this number, the response
      message will contain a valid value in the `next_page_token` field. The
      default value is 20, and the maximum page size is 100.
    pageToken: Optional. A page token to request the next page of results. You
      get the token from the `next_page_token` field of the response from the
      previous call.
    parent: Required. The name of the project for which to list jobs.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)