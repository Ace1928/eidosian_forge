from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3QueryInput(_messages.Message):
    """Represents the query input. It can contain one of: 1. A conversational
  query in the form of text. 2. An intent query that specifies which intent to
  trigger. 3. Natural language speech audio to be processed. 4. An event to be
  triggered. 5. DTMF digits to invoke an intent and fill in parameter value.
  6. The results of a tool executed by the client.

  Fields:
    audio: The natural language speech audio to be processed.
    dtmf: The DTMF event to be handled.
    event: The event to be triggered.
    intent: The intent to be triggered.
    languageCode: Required. The language of the input. See [Language
      Support](https://cloud.google.com/dialogflow/cx/docs/reference/language)
      for a list of the currently supported language codes. Note that queries
      in the same session do not necessarily need to specify the same
      language.
    text: The natural language text to be processed.
  """
    audio = _messages.MessageField('GoogleCloudDialogflowCxV3AudioInput', 1)
    dtmf = _messages.MessageField('GoogleCloudDialogflowCxV3DtmfInput', 2)
    event = _messages.MessageField('GoogleCloudDialogflowCxV3EventInput', 3)
    intent = _messages.MessageField('GoogleCloudDialogflowCxV3IntentInput', 4)
    languageCode = _messages.StringField(5)
    text = _messages.MessageField('GoogleCloudDialogflowCxV3TextInput', 6)