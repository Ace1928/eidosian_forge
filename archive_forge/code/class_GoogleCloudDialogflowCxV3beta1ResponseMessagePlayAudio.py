from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ResponseMessagePlayAudio(_messages.Message):
    """Specifies an audio clip to be played by the client as part of the
  response.

  Fields:
    allowPlaybackInterruption: Output only. Whether the playback of this
      message can be interrupted by the end user's speech and the client can
      then starts the next Dialogflow request.
    audioUri: Required. URI of the audio clip. Dialogflow does not impose any
      validation on this value. It is specific to the client that reads it.
  """
    allowPlaybackInterruption = _messages.BooleanField(1)
    audioUri = _messages.StringField(2)