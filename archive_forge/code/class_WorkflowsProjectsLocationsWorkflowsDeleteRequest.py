from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowsProjectsLocationsWorkflowsDeleteRequest(_messages.Message):
    """A WorkflowsProjectsLocationsWorkflowsDeleteRequest object.

  Fields:
    name: Required. Name of the workflow to be deleted. Format:
      projects/{project}/locations/{location}/workflows/{workflow}
  """
    name = _messages.StringField(1, required=True)