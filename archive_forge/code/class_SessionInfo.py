from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SessionInfo(_messages.Message):
    """[Preview] Information related to sessions.

  Fields:
    sessionId: Output only. The id of the session.
  """
    sessionId = _messages.StringField(1)