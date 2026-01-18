from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ElementaryStream(_messages.Message):
    """Encoding of an input file such as an audio, video, or text track.
  Elementary streams must be packaged before mapping and sharing between
  different output formats.

  Fields:
    audioStream: Encoding of an audio stream.
    key: A unique key for this elementary stream.
    textStream: Encoding of a text stream. For example, closed captions or
      subtitles.
    videoStream: Encoding of a video stream.
  """
    audioStream = _messages.MessageField('AudioStream', 1)
    key = _messages.StringField(2)
    textStream = _messages.MessageField('TextStream', 3)
    videoStream = _messages.MessageField('VideoStream', 4)