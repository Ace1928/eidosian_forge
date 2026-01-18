from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsRecognizersRecognizeRequest(_messages.Message):
    """A SpeechProjectsLocationsRecognizersRecognizeRequest object.

  Fields:
    recognizeRequest: A RecognizeRequest resource to be passed as the request
      body.
    recognizer: Required. The name of the Recognizer to use during
      recognition. The expected format is
      `projects/{project}/locations/{location}/recognizers/{recognizer}`. The
      {recognizer} segment may be set to `_` to use an empty implicit
      Recognizer.
  """
    recognizeRequest = _messages.MessageField('RecognizeRequest', 1)
    recognizer = _messages.StringField(2, required=True)