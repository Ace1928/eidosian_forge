from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Romanization(_messages.Message):
    """A single romanization response.

  Fields:
    detectedLanguageCode: The ISO-639 language code of source text in the
      initial request, detected automatically, if no source language was
      passed within the initial request. If the source language was passed,
      auto-detection of the language does not occur and this field is empty.
    romanizedText: Romanized text. If an error occurs during romanization,
      this field might be excluded from the response.
  """
    detectedLanguageCode = _messages.StringField(1)
    romanizedText = _messages.StringField(2)