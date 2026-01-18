from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1AnnotateVideoRequest(_messages.Message):
    """Video annotation request.

  Enums:
    FeaturesValueListEntryValuesEnum:

  Fields:
    features: Required. Requested video annotation features.
    inputContent: The video data bytes. If unset, the input video(s) should be
      specified via the `input_uri`. If set, `input_uri` must be unset.
    inputUri: Input video location. Currently, only [Cloud
      Storage](https://cloud.google.com/storage/) URIs are supported. URIs
      must be specified in the following format: `gs://bucket-id/object-id`
      (other URI formats return google.rpc.Code.INVALID_ARGUMENT). For more
      information, see [Request
      URIs](https://cloud.google.com/storage/docs/request-endpoints). To
      identify multiple videos, a video URI may include wildcards in the
      `object-id`. Supported wildcards: '*' to match 0 or more characters; '?'
      to match 1 character. If unset, the input video should be embedded in
      the request as `input_content`. If set, `input_content` must be unset.
    locationId: Optional. Cloud region where annotation should take place.
      Supported cloud regions are: `us-east1`, `us-west1`, `europe-west1`,
      `asia-east1`. If no region is specified, the region will be determined
      based on video file location.
    outputUri: Optional. Location where the output (in JSON format) should be
      stored. Currently, only [Cloud
      Storage](https://cloud.google.com/storage/) URIs are supported. These
      must be specified in the following format: `gs://bucket-id/object-id`
      (other URI formats return google.rpc.Code.INVALID_ARGUMENT). For more
      information, see [Request
      URIs](https://cloud.google.com/storage/docs/request-endpoints).
    videoContext: Additional video context and/or feature-specific parameters.
  """

    class FeaturesValueListEntryValuesEnum(_messages.Enum):
        """FeaturesValueListEntryValuesEnum enum type.

    Values:
      FEATURE_UNSPECIFIED: Unspecified.
      LABEL_DETECTION: Label detection. Detect objects, such as dog or flower.
      SHOT_CHANGE_DETECTION: Shot change detection.
      EXPLICIT_CONTENT_DETECTION: Explicit content detection.
      FACE_DETECTION: Human face detection.
      SPEECH_TRANSCRIPTION: Speech transcription.
      TEXT_DETECTION: OCR text detection and tracking.
      OBJECT_TRACKING: Object detection and tracking.
      LOGO_RECOGNITION: Logo detection, tracking, and recognition.
      PERSON_DETECTION: Person detection.
    """
        FEATURE_UNSPECIFIED = 0
        LABEL_DETECTION = 1
        SHOT_CHANGE_DETECTION = 2
        EXPLICIT_CONTENT_DETECTION = 3
        FACE_DETECTION = 4
        SPEECH_TRANSCRIPTION = 5
        TEXT_DETECTION = 6
        OBJECT_TRACKING = 7
        LOGO_RECOGNITION = 8
        PERSON_DETECTION = 9
    features = _messages.EnumField('FeaturesValueListEntryValuesEnum', 1, repeated=True)
    inputContent = _messages.BytesField(2)
    inputUri = _messages.StringField(3)
    locationId = _messages.StringField(4)
    outputUri = _messages.StringField(5)
    videoContext = _messages.MessageField('GoogleCloudVideointelligenceV1VideoContext', 6)