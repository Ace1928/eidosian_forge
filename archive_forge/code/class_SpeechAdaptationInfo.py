from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechAdaptationInfo(_messages.Message):
    """Information on speech adaptation use in results

  Fields:
    adaptationTimeout: Whether there was a timeout when applying speech
      adaptation. If true, adaptation had no effect in the response
      transcript.
    timeoutMessage: If set, returns a message specifying which part of the
      speech adaptation request timed out.
  """
    adaptationTimeout = _messages.BooleanField(1)
    timeoutMessage = _messages.StringField(2)