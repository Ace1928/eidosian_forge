from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootTranslationRequestInfo(_messages.Message):
    """Each TranslationRequestInfo corresponds to a request sent to the
  translation server.

  Fields:
    detectedLanguageCodes: The ISO-639 language code of source text in the
      initial request, detected automatically, if no source language was
      passed within the initial request. If the source language was passed,
      auto-detection of the language does not occur and this field is empty.
    totalContentSize: The sum of the size of all the contents in the request.
  """
    detectedLanguageCodes = _messages.StringField(1, repeated=True)
    totalContentSize = _messages.IntegerField(2)