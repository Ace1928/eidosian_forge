from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionImage(_messages.Message):
    """Image.

  Fields:
    encoding: Image encoding, encoded as "image/png" or "image/jpg".
    image: Raw bytes.
    imageRaiScores: RAI scores for generated image.
    raiInfo: RAI info for image.
    semanticFilterResponse: Semantic filter info for image.
    text: Text/Expanded text input for imagen.
    uri: Path to another storage (typically Google Cloud Storage).
  """
    encoding = _messages.StringField(1)
    image = _messages.BytesField(2)
    imageRaiScores = _messages.MessageField('CloudAiLargeModelsVisionImageRAIScores', 3)
    raiInfo = _messages.MessageField('CloudAiLargeModelsVisionRaiInfo', 4)
    semanticFilterResponse = _messages.MessageField('CloudAiLargeModelsVisionSemanticFilterResponse', 5)
    text = _messages.StringField(6)
    uri = _messages.StringField(7)