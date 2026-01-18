from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1LocalizedObjectAnnotation(_messages.Message):
    """Set of detected objects with bounding boxes.

  Fields:
    boundingPoly: Image region to which this object belongs. This must be
      populated.
    languageCode: The BCP-47 language code, such as "en-US" or "sr-Latn". For
      more information, see
      http://www.unicode.org/reports/tr35/#Unicode_locale_identifier.
    mid: Object ID that should align with EntityAnnotation mid.
    name: Object name, expressed in its `language_code` language.
    score: Score of the result. Range [0, 1].
  """
    boundingPoly = _messages.MessageField('GoogleCloudVisionV1p4beta1BoundingPoly', 1)
    languageCode = _messages.StringField(2)
    mid = _messages.StringField(3)
    name = _messages.StringField(4)
    score = _messages.FloatField(5, variant=_messages.Variant.FLOAT)