from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p1beta1AnnotateVideoProgress(_messages.Message):
    """Video annotation progress. Included in the `metadata` field of the
  `Operation` returned by the `GetOperation` call of the
  `google::longrunning::Operations` service.

  Fields:
    annotationProgress: Progress metadata for all videos specified in
      `AnnotateVideoRequest`.
  """
    annotationProgress = _messages.MessageField('GoogleCloudVideointelligenceV1p1beta1VideoAnnotationProgress', 1, repeated=True)