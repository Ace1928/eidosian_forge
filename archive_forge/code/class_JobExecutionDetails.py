from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobExecutionDetails(_messages.Message):
    """Information about the execution of a job.

  Fields:
    nextPageToken: If present, this response does not contain all requested
      tasks. To obtain the next page of results, repeat the request with
      page_token set to this value.
    stages: The stages of the job execution.
  """
    nextPageToken = _messages.StringField(1)
    stages = _messages.MessageField('StageSummary', 2, repeated=True)