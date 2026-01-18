from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerflexProjectsLocationsWorkflowsGetRequest(_messages.Message):
    """A ComposerflexProjectsLocationsWorkflowsGetRequest object.

  Fields:
    name: The resource name of the workflow to retrieve. Must be in the form
      "projects/{projectId}/locations/{locationId}/workflows/{workflowId}."
  """
    name = _messages.StringField(1, required=True)