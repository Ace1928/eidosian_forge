from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerflexProjectsLocationsWorkflowsRunsGetRequest(_messages.Message):
    """A ComposerflexProjectsLocationsWorkflowsRunsGetRequest object.

  Fields:
    name: The resource name for the WorkflowRun in the form: "projects/{projec
      tId}/locations/{locationId}/workflows/{workflowId}/runs/{runId}".
  """
    name = _messages.StringField(1, required=True)