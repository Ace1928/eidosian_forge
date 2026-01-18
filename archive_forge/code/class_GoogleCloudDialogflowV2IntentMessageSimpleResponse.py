from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageSimpleResponse(_messages.Message):
    """The simple response message containing speech or text.

  Fields:
    displayText: Optional. The text to display.
    ssml: One of text_to_speech or ssml must be provided. Structured spoken
      response to the user in the SSML format. Mutually exclusive with
      text_to_speech.
    textToSpeech: One of text_to_speech or ssml must be provided. The plain
      text of the speech output. Mutually exclusive with ssml.
  """
    displayText = _messages.StringField(1)
    ssml = _messages.StringField(2)
    textToSpeech = _messages.StringField(3)