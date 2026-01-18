from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListDebuggeesResponse(_messages.Message):
    """Response for listing debuggees.

  Fields:
    debuggees: List of debuggees accessible to the calling user. The fields
      `debuggee.id` and `description` are guaranteed to be set. The
      `description` field is a human readable field provided by agents and can
      be displayed to users.
  """
    debuggees = _messages.MessageField('Debuggee', 1, repeated=True)