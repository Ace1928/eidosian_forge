from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InlineResult(_messages.Message):
    """Final results returned inline in the recognition response.

  Fields:
    srtCaptions: The transcript for the audio file as SRT formatted captions.
      This is populated only when `SRT` output is requested.
    transcript: The transcript for the audio file.
    vttCaptions: The transcript for the audio file as VTT formatted captions.
      This is populated only when `VTT` output is requested.
  """
    srtCaptions = _messages.StringField(1)
    transcript = _messages.MessageField('BatchRecognizeResults', 2)
    vttCaptions = _messages.StringField(3)