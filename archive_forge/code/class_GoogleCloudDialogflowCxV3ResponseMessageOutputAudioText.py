from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ResponseMessageOutputAudioText(_messages.Message):
    """A text or ssml response that is preferentially used for TTS output audio
  synthesis, as described in the comment on the ResponseMessage message.

  Fields:
    allowPlaybackInterruption: Output only. Whether the playback of this
      message can be interrupted by the end user's speech and the client can
      then starts the next Dialogflow request.
    ssml: The SSML text to be synthesized. For more information, see
      [SSML](/speech/text-to-speech/docs/ssml).
    text: The raw text to be synthesized.
  """
    allowPlaybackInterruption = _messages.BooleanField(1)
    ssml = _messages.StringField(2)
    text = _messages.StringField(3)