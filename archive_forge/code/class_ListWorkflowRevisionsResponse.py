from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkflowRevisionsResponse(_messages.Message):
    """Response for the ListWorkflowRevisions method.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workflows: The revisions of the workflow, ordered in reverse chronological
      order.
  """
    nextPageToken = _messages.StringField(1)
    workflows = _messages.MessageField('Workflow', 2, repeated=True)