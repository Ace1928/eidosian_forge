from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TestingProjectsDeviceSessionsPatchRequest(_messages.Message):
    """A TestingProjectsDeviceSessionsPatchRequest object.

  Fields:
    deviceSession: A DeviceSession resource to be passed as the request body.
    name: Optional. Name of the DeviceSession, e.g.
      "projects/{project_id}/deviceSessions/{session_id}"
    updateMask: Required. The list of fields to update.
  """
    deviceSession = _messages.MessageField('DeviceSession', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)