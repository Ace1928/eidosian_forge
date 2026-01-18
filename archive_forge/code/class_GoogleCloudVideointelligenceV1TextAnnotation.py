from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1TextAnnotation(_messages.Message):
    """Annotations related to one detected OCR text snippet. This will contain
  the corresponding text, confidence value, and frame level information for
  each detection.

  Fields:
    segments: All video segments where OCR detected text appears.
    text: The detected text.
    version: Feature version.
  """
    segments = _messages.MessageField('GoogleCloudVideointelligenceV1TextSegment', 1, repeated=True)
    text = _messages.StringField(2)
    version = _messages.StringField(3)