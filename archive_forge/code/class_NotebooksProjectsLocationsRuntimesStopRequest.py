from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesStopRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesStopRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
    stopRuntimeRequest: A StopRuntimeRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    stopRuntimeRequest = _messages.MessageField('StopRuntimeRequest', 2)