from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowsProjectsLocationsWorkflowsCreateRequest(_messages.Message):
    """A WorkflowsProjectsLocationsWorkflowsCreateRequest object.

  Fields:
    parent: Required. Project and location in which the workflow should be
      created. Format: projects/{project}/locations/{location}
    workflow: A Workflow resource to be passed as the request body.
    workflowId: Required. The ID of the workflow to be created. It has to
      fulfill the following requirements: * Must contain only letters,
      numbers, underscores and hyphens. * Must start with a letter. * Must be
      between 1-64 characters. * Must end with a number or a letter. * Must be
      unique within the customer project and location.
  """
    parent = _messages.StringField(1, required=True)
    workflow = _messages.MessageField('Workflow', 2)
    workflowId = _messages.StringField(3)