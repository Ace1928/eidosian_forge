from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. User-defined unique ID of this instance.
    parent: Required. Format:
      `parent=projects/{project_id}/locations/{location}`
    requestId: Optional. Idempotent request UUID.
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)