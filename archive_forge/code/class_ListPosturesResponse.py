from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPosturesResponse(_messages.Message):
    """Message for response to listing Postures.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    postures: The list of Posture.
    unreachable: Unreachable resources.
  """
    nextPageToken = _messages.StringField(1)
    postures = _messages.MessageField('Posture', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)