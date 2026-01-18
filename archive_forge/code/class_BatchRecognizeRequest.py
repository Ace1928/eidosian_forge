from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchRecognizeRequest(_messages.Message):
    """Request message for the BatchRecognize method.

  Enums:
    ProcessingStrategyValueValuesEnum: Processing strategy to use for this
      request.

  Fields:
    config: Features and audio metadata to use for the Automatic Speech
      Recognition. This field in combination with the config_mask field can be
      used to override parts of the default_recognition_config of the
      Recognizer resource.
    configMask: The list of fields in config that override the values in the
      default_recognition_config of the recognizer during this recognition
      request. If no mask is provided, all given fields in config override the
      values in the recognizer for this recognition request. If a mask is
      provided, only the fields listed in the mask override the config in the
      recognizer for this recognition request. If a wildcard (`*`) is
      provided, config completely overrides and replaces the config in the
      recognizer for this recognition request.
    files: Audio files with file metadata for ASR. The maximum number of files
      allowed to be specified is 5.
    processingStrategy: Processing strategy to use for this request.
    recognitionOutputConfig: Configuration options for where to output the
      transcripts of each file.
    recognizer: Required. The name of the Recognizer to use during
      recognition. The expected format is
      `projects/{project}/locations/{location}/recognizers/{recognizer}`. The
      {recognizer} segment may be set to `_` to use an empty implicit
      Recognizer.
  """

    class ProcessingStrategyValueValuesEnum(_messages.Enum):
        """Processing strategy to use for this request.

    Values:
      PROCESSING_STRATEGY_UNSPECIFIED: Default value for the processing
        strategy. The request is processed as soon as its received.
      DYNAMIC_BATCHING: If selected, processes the request during lower
        utilization periods for a price discount. The request is fulfilled
        within 24 hours.
    """
        PROCESSING_STRATEGY_UNSPECIFIED = 0
        DYNAMIC_BATCHING = 1
    config = _messages.MessageField('RecognitionConfig', 1)
    configMask = _messages.StringField(2)
    files = _messages.MessageField('BatchRecognizeFileMetadata', 3, repeated=True)
    processingStrategy = _messages.EnumField('ProcessingStrategyValueValuesEnum', 4)
    recognitionOutputConfig = _messages.MessageField('RecognitionOutputConfig', 5)
    recognizer = _messages.StringField(6)