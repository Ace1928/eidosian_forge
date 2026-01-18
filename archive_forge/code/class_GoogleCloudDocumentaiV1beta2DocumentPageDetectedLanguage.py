from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageDetectedLanguage(_messages.Message):
    """Detected language for a structural component.

  Fields:
    confidence: Confidence of detected language. Range `[0, 1]`.
    languageCode: The [BCP-47 language
      code](https://www.unicode.org/reports/tr35/#Unicode_locale_identifier),
      such as `en-US` or `sr-Latn`.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    languageCode = _messages.StringField(2)