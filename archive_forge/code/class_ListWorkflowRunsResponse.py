from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkflowRunsResponse(_messages.Message):
    """Response for ListWorkflowRunsRequest.

  Fields:
    nextPageToken: The page token used to query for the next page if one
      exists.
    workflowRuns: The returned list of WorkflowRuns.
  """
    nextPageToken = _messages.StringField(1)
    workflowRuns = _messages.MessageField('WorkflowRun', 2, repeated=True)