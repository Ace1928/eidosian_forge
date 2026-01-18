from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListDeviceSessionsResponse(_messages.Message):
    """A list of device sessions.

  Fields:
    deviceSessions: The sessions matching the specified filter in the given
      cloud project.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    deviceSessions = _messages.MessageField('DeviceSession', 1, repeated=True)
    nextPageToken = _messages.StringField(2)