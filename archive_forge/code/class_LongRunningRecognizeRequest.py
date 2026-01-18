from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LongRunningRecognizeRequest(_messages.Message):
    """The top-level message sent by the client for the `LongRunningRecognize`
  method.

  Fields:
    audio: Required. The audio data to be recognized.
    config: Required. Provides information to the recognizer that specifies
      how to process the request.
    outputConfig: Optional. Specifies an optional destination for the
      recognition results.
  """
    audio = _messages.MessageField('RecognitionAudio', 1)
    config = _messages.MessageField('RecognitionConfig', 2)
    outputConfig = _messages.MessageField('TranscriptOutputConfig', 3)