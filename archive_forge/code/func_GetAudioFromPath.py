from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from six.moves import urllib
def GetAudioFromPath(path):
    """Determine whether path to audio is local, build RecognitionAudio message.

    Args:
      path: str, the path to the audio.

    Raises:
      AudioException: If audio is not found locally and does not appear to be
        Google Cloud Storage URL.

    Returns:
      speech_v1_messages.RecognitionAudio, the audio message.
    """
    return GetRecognitionAudioFromPath(path, version)