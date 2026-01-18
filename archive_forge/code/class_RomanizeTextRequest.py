from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RomanizeTextRequest(_messages.Message):
    """The request message for synchronous romanization.

  Fields:
    contents: Required. The content of the input in string format.
    sourceLanguageCode: Optional. The ISO-639 language code of the input text
      if known, for example, "hi" or "zh". If the source language isn't
      specified, the API attempts to identify the source language
      automatically and returns the source language for each content in the
      response.
  """
    contents = _messages.StringField(1, repeated=True)
    sourceLanguageCode = _messages.StringField(2)