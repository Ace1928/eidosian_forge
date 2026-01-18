from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3BargeInConfig(_messages.Message):
    """Configuration of the barge-in behavior. Barge-in instructs the API to
  return a detected utterance at a proper time while the client is playing
  back the response audio from a previous request. When the client sees the
  utterance, it should stop the playback and immediately get ready for
  receiving the responses for the current request. The barge-in handling
  requires the client to start streaming audio input as soon as it starts
  playing back the audio from the previous response. The playback is modeled
  into two phases: * No barge-in phase: which goes first and during which
  speech detection should not be carried out. * Barge-in phase: which follows
  the no barge-in phase and during which the API starts speech detection and
  may inform the client that an utterance has been detected. Note that no-
  speech event is not expected in this phase. The client provides this
  configuration in terms of the durations of those two phases. The durations
  are measured in terms of the audio length from the the start of the input
  audio. No-speech event is a response with END_OF_UTTERANCE without any
  transcript following up.

  Fields:
    noBargeInDuration: Duration that is not eligible for barge-in at the
      beginning of the input audio.
    totalDuration: Total duration for the playback at the beginning of the
      input audio.
  """
    noBargeInDuration = _messages.StringField(1)
    totalDuration = _messages.StringField(2)