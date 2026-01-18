from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesGetConfigRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesGetConfigRequest object.

  Fields:
    name: Required. Format: `projects/{project_id}/locations/{location}`
  """
    name = _messages.StringField(1, required=True)