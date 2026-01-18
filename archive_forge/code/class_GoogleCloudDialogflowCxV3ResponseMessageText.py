from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ResponseMessageText(_messages.Message):
    """The text response message.

  Fields:
    allowPlaybackInterruption: Output only. Whether the playback of this
      message can be interrupted by the end user's speech and the client can
      then starts the next Dialogflow request.
    text: Required. A collection of text responses.
  """
    allowPlaybackInterruption = _messages.BooleanField(1)
    text = _messages.StringField(2, repeated=True)