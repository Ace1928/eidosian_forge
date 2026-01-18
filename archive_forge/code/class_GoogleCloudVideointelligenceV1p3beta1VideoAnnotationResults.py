from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1VideoAnnotationResults(_messages.Message):
    """Annotation results for a single video.

  Fields:
    celebrityRecognitionAnnotations: Celebrity recognition annotations.
    error: If set, indicates an error. Note that for a single
      `AnnotateVideoRequest` some videos may succeed and some may fail.
    explicitAnnotation: Explicit content annotation.
    faceAnnotations: Deprecated. Please use `face_detection_annotations`
      instead.
    faceDetectionAnnotations: Face detection annotations.
    frameLabelAnnotations: Label annotations on frame level. There is exactly
      one element for each unique label.
    inputUri: Video file location in [Cloud
      Storage](https://cloud.google.com/storage/).
    logoRecognitionAnnotations: Annotations for list of logos detected,
      tracked and recognized in video.
    objectAnnotations: Annotations for list of objects detected and tracked in
      video.
    personDetectionAnnotations: Person detection annotations.
    segment: Video segment on which the annotation is run.
    segmentLabelAnnotations: Topical label annotations on video level or user-
      specified segment level. There is exactly one element for each unique
      label.
    segmentPresenceLabelAnnotations: Presence label annotations on video level
      or user-specified segment level. There is exactly one element for each
      unique label. Compared to the existing topical
      `segment_label_annotations`, this field presents more fine-grained,
      segment-level labels detected in video content and is made available
      only when the client sets `LabelDetectionConfig.model` to
      "builtin/latest" in the request.
    shotAnnotations: Shot annotations. Each shot is represented as a video
      segment.
    shotLabelAnnotations: Topical label annotations on shot level. There is
      exactly one element for each unique label.
    shotPresenceLabelAnnotations: Presence label annotations on shot level.
      There is exactly one element for each unique label. Compared to the
      existing topical `shot_label_annotations`, this field presents more
      fine-grained, shot-level labels detected in video content and is made
      available only when the client sets `LabelDetectionConfig.model` to
      "builtin/latest" in the request.
    speechTranscriptions: Speech transcription.
    textAnnotations: OCR text detection and tracking. Annotations for list of
      detected text snippets. Each will have list of frame information
      associated with it.
  """
    celebrityRecognitionAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1CelebrityRecognitionAnnotation', 1)
    error = _messages.MessageField('GoogleRpcStatus', 2)
    explicitAnnotation = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1ExplicitContentAnnotation', 3)
    faceAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1FaceAnnotation', 4, repeated=True)
    faceDetectionAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1FaceDetectionAnnotation', 5, repeated=True)
    frameLabelAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LabelAnnotation', 6, repeated=True)
    inputUri = _messages.StringField(7)
    logoRecognitionAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LogoRecognitionAnnotation', 8, repeated=True)
    objectAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1ObjectTrackingAnnotation', 9, repeated=True)
    personDetectionAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1PersonDetectionAnnotation', 10, repeated=True)
    segment = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1VideoSegment', 11)
    segmentLabelAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LabelAnnotation', 12, repeated=True)
    segmentPresenceLabelAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LabelAnnotation', 13, repeated=True)
    shotAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1VideoSegment', 14, repeated=True)
    shotLabelAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LabelAnnotation', 15, repeated=True)
    shotPresenceLabelAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1LabelAnnotation', 16, repeated=True)
    speechTranscriptions = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1SpeechTranscription', 17, repeated=True)
    textAnnotations = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1TextAnnotation', 18, repeated=True)