from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesCheckUpgradabilityRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesCheckUpgradabilityRequest object.

  Fields:
    notebookInstance: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
  """
    notebookInstance = _messages.StringField(1, required=True)