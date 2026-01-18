from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesDeleteRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesDeleteRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
    requestId: Idempotent request UUID.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)