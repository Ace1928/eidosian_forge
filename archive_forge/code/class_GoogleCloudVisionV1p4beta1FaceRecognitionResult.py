from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1FaceRecognitionResult(_messages.Message):
    """Information about a face's identity.

  Fields:
    celebrity: The Celebrity that this face was matched to.
    confidence: Recognition confidence. Range [0, 1].
  """
    celebrity = _messages.MessageField('GoogleCloudVisionV1p4beta1Celebrity', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)