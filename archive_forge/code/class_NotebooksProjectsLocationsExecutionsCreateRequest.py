from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsExecutionsCreateRequest(_messages.Message):
    """A NotebooksProjectsLocationsExecutionsCreateRequest object.

  Fields:
    execution: A Execution resource to be passed as the request body.
    executionId: Required. User-defined unique ID of this execution.
    parent: Required. Format:
      `parent=projects/{project_id}/locations/{location}`
  """
    execution = _messages.MessageField('Execution', 1)
    executionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)