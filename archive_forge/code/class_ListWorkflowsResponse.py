from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkflowsResponse(_messages.Message):
    """Response to ListWorkflowsRequest.

  Fields:
    nextPageToken: The page token used to query for the next page if one
      exists.
    workflows: The list of workflows returned.
  """
    nextPageToken = _messages.StringField(1)
    workflows = _messages.MessageField('Workflow', 2, repeated=True)