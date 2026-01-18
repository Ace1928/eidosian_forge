from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkflowsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkflowsCreateRequest object.

  Fields:
    parent: Required. Format: `projects/{project}/locations/{location}`
    validateOnly: When true, the query is validated only, but not executed.
    workflow: A Workflow resource to be passed as the request body.
    workflowId: Required. The ID to use for the Workflow, which will become
      the final component of the Workflow's resource name.
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    workflow = _messages.MessageField('Workflow', 3)
    workflowId = _messages.StringField(4)