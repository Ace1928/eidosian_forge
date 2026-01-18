from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1ProductSearchResultsObjectAnnotation(_messages.Message):
    """Prediction for what the object in the bounding box is.

  Fields:
    languageCode: The BCP-47 language code, such as "en-US" or "sr-Latn". For
      more information, see
      http://www.unicode.org/reports/tr35/#Unicode_locale_identifier.
    mid: Object ID that should align with EntityAnnotation mid.
    name: Object name, expressed in its `language_code` language.
    score: Score of the result. Range [0, 1].
  """
    languageCode = _messages.StringField(1)
    mid = _messages.StringField(2)
    name = _messages.StringField(3)
    score = _messages.FloatField(4, variant=_messages.Variant.FLOAT)