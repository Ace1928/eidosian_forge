from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesDeleteRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesDeleteRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    requestId: Optional. Idempotent request UUID.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)