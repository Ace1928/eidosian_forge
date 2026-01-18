from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateSessionsResponse(_messages.Message):
    """The response for BatchCreateSessions.

  Fields:
    session: The freshly created sessions.
  """
    session = _messages.MessageField('Session', 1, repeated=True)