from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3AudioInput(_messages.Message):
    """Represents the natural speech audio to be processed.

  Fields:
    audio: The natural language speech audio to be processed. A single request
      can contain up to 2 minutes of speech audio data. The transcribed text
      cannot contain more than 256 bytes. For non-streaming audio detect
      intent, both `config` and `audio` must be provided. For streaming audio
      detect intent, `config` must be provided in the first request and
      `audio` must be provided in all following requests.
    config: Required. Instructs the speech recognizer how to process the
      speech audio.
  """
    audio = _messages.BytesField(1)
    config = _messages.MessageField('GoogleCloudDialogflowCxV3InputAudioConfig', 2)