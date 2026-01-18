from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v2 import speech_v2_messages as messages
def BatchRecognize(self, request, global_params=None):
    """Performs batch asynchronous speech recognition: send a request with N audio files and receive a long running operation that can be polled to see when the transcriptions are finished.

      Args:
        request: (BatchRecognizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('BatchRecognize')
    return self._RunMethod(config, request, global_params=global_params)