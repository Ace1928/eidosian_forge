from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetectLanguageResponse(_messages.Message):
    """The response message for language detection.

  Fields:
    languages: A list of detected languages sorted by detection confidence in
      descending order. The most probable language first.
  """
    languages = _messages.MessageField('DetectedLanguage', 1, repeated=True)