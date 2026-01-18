from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchRecognizeFileResult(_messages.Message):
    """Final results for a single file.

  Fields:
    cloudStorageResult: Recognition results written to Cloud Storage. This is
      populated only when GcsOutputConfig is set in the
      RecognitionOutputConfig.
    error: Error if one was encountered.
    inlineResult: Recognition results. This is populated only when
      InlineOutputConfig is set in the RecognitionOutputConfig.
    metadata: A RecognitionResponseMetadata attribute.
    transcript: Deprecated. Use `inline_result.transcript` instead.
    uri: Deprecated. Use `cloud_storage_result.native_format_uri` instead.
  """
    cloudStorageResult = _messages.MessageField('CloudStorageResult', 1)
    error = _messages.MessageField('Status', 2)
    inlineResult = _messages.MessageField('InlineResult', 3)
    metadata = _messages.MessageField('RecognitionResponseMetadata', 4)
    transcript = _messages.MessageField('BatchRecognizeResults', 5)
    uri = _messages.StringField(6)