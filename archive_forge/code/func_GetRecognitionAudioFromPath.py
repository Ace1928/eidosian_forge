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
def GetRecognitionAudioFromPath(path, version):
    """Determine whether path to audio is local, set RecognitionAudio message."""
    messages = apis.GetMessagesModule(SPEECH_API, version)
    audio = messages.RecognitionAudio()
    if os.path.isfile(path):
        audio.content = files.ReadBinaryFileContents(path)
    elif storage_util.ObjectReference.IsStorageUrl(path):
        audio.uri = path
    else:
        raise AudioException('Invalid audio source [{}]. The source must either be a local path or a Google Cloud Storage URL (such as gs://bucket/object).'.format(path))
    return audio