from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetectedLanguage(_messages.Message):
    """The response message for language detection.

  Fields:
    confidence: The confidence of the detection result for this language.
    languageCode: The BCP-47 language code of source content in the request,
      detected automatically.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    languageCode = _messages.StringField(2)