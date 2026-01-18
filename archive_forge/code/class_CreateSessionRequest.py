from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateSessionRequest(_messages.Message):
    """The request for CreateSession.

  Fields:
    session: Required. The session to create.
  """
    session = _messages.MessageField('Session', 1)