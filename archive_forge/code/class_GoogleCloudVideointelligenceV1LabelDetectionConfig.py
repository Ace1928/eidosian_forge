from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1LabelDetectionConfig(_messages.Message):
    """Config for LABEL_DETECTION.

  Enums:
    LabelDetectionModeValueValuesEnum: What labels should be detected with
      LABEL_DETECTION, in addition to video-level labels or segment-level
      labels. If unspecified, defaults to `SHOT_MODE`.

  Fields:
    frameConfidenceThreshold: The confidence threshold we perform filtering on
      the labels from frame-level detection. If not set, it is set to 0.4 by
      default. The valid range for this threshold is [0.1, 0.9]. Any value set
      outside of this range will be clipped. Note: For best results, follow
      the default threshold. We will update the default threshold everytime
      when we release a new model.
    labelDetectionMode: What labels should be detected with LABEL_DETECTION,
      in addition to video-level labels or segment-level labels. If
      unspecified, defaults to `SHOT_MODE`.
    model: Model to use for label detection. Supported values:
      "builtin/stable" (the default if unset) and "builtin/latest".
    stationaryCamera: Whether the video has been shot from a stationary (i.e.,
      non-moving) camera. When set to true, might improve detection accuracy
      for moving objects. Should be used with `SHOT_AND_FRAME_MODE` enabled.
    videoConfidenceThreshold: The confidence threshold we perform filtering on
      the labels from video-level and shot-level detections. If not set, it's
      set to 0.3 by default. The valid range for this threshold is [0.1, 0.9].
      Any value set outside of this range will be clipped. Note: For best
      results, follow the default threshold. We will update the default
      threshold everytime when we release a new model.
  """

    class LabelDetectionModeValueValuesEnum(_messages.Enum):
        """What labels should be detected with LABEL_DETECTION, in addition to
    video-level labels or segment-level labels. If unspecified, defaults to
    `SHOT_MODE`.

    Values:
      LABEL_DETECTION_MODE_UNSPECIFIED: Unspecified.
      SHOT_MODE: Detect shot-level labels.
      FRAME_MODE: Detect frame-level labels.
      SHOT_AND_FRAME_MODE: Detect both shot-level and frame-level labels.
    """
        LABEL_DETECTION_MODE_UNSPECIFIED = 0
        SHOT_MODE = 1
        FRAME_MODE = 2
        SHOT_AND_FRAME_MODE = 3
    frameConfidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    labelDetectionMode = _messages.EnumField('LabelDetectionModeValueValuesEnum', 2)
    model = _messages.StringField(3)
    stationaryCamera = _messages.BooleanField(4)
    videoConfidenceThreshold = _messages.FloatField(5, variant=_messages.Variant.FLOAT)