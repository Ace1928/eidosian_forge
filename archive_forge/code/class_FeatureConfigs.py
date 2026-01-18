from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureConfigs(_messages.Message):
    """FeatureConfigs configure different IMS properties.

  Fields:
    speechTranscriptionConfig: Configure transcription options for speech:
      keyword.
  """
    speechTranscriptionConfig = _messages.MessageField('SpeechTranscriptionConfig', 1)