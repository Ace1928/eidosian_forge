from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowsProjectsLocationsWorkflowsGetRequest(_messages.Message):
    """A WorkflowsProjectsLocationsWorkflowsGetRequest object.

  Fields:
    name: Required. Name of the workflow which information should be
      retrieved. Format:
      projects/{project}/locations/{location}/workflows/{workflow}
  """
    name = _messages.StringField(1, required=True)