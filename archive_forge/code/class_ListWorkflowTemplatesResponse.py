from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkflowTemplatesResponse(_messages.Message):
    """A response to a request to list workflow templates in a project.

  Fields:
    nextPageToken: Output only. This token is included in the response if
      there are more results to fetch. To fetch additional results, provide
      this value as the page_token in a subsequent
      ListWorkflowTemplatesRequest.
    templates: Output only. WorkflowTemplates list.
    unreachable: Output only. List of workflow templates that could not be
      included in the response. Attempting to get one of these resources may
      indicate why it was not included in the list response.
  """
    nextPageToken = _messages.StringField(1)
    templates = _messages.MessageField('WorkflowTemplate', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)