from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageTelephonySynthesizeSpeech(_messages.Message):
    """Synthesizes speech and plays back the synthesized audio to the caller in
  Telephony Gateway. Telephony Gateway takes the synthesizer settings from
  `DetectIntentResponse.output_audio_config` which can either be set at
  request-level or can come from the agent-level synthesizer config.

  Fields:
    ssml: The SSML to be synthesized. For more information, see
      [SSML](https://developers.google.com/actions/reference/ssml).
    text: The raw text to be synthesized.
  """
    ssml = _messages.StringField(1)
    text = _messages.StringField(2)