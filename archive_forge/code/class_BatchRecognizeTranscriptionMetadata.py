from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchRecognizeTranscriptionMetadata(_messages.Message):
    """Metadata about transcription for a single file (for example, progress
  percent).

  Fields:
    error: Error if one was encountered.
    progressPercent: How much of the file has been transcribed so far.
    uri: The Cloud Storage URI to which recognition results will be written.
  """
    error = _messages.MessageField('Status', 1)
    progressPercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    uri = _messages.StringField(3)