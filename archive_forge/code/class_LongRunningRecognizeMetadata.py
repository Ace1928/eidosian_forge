from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LongRunningRecognizeMetadata(_messages.Message):
    """Describes the progress of a long-running `LongRunningRecognize` call. It
  is included in the `metadata` field of the `Operation` returned by the
  `GetOperation` call of the `google::longrunning::Operations` service.

  Fields:
    lastUpdateTime: Time of the most recent processing update.
    progressPercent: Approximate percentage of audio processed thus far.
      Guaranteed to be 100 when the audio is fully processed and the results
      are available.
    startTime: Time when the request was received.
    uri: Output only. The URI of the audio file being transcribed. Empty if
      the audio was sent as byte content.
  """
    lastUpdateTime = _messages.StringField(1)
    progressPercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(3)
    uri = _messages.StringField(4)