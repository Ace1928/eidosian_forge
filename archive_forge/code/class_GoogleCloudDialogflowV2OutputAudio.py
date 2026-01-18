from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2OutputAudio(_messages.Message):
    """Represents the natural language speech audio to be played to the end
  user.

  Fields:
    audio: The natural language speech audio.
    config: Instructs the speech synthesizer how to generate the speech audio.
  """
    audio = _messages.BytesField(1)
    config = _messages.MessageField('GoogleCloudDialogflowV2OutputAudioConfig', 2)