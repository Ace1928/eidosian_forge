from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1beta2AnnotateVideoResponse(_messages.Message):
    """Video annotation response. Included in the `response` field of the
  `Operation` returned by the `GetOperation` call of the
  `google::longrunning::Operations` service.

  Fields:
    annotationResults: Annotation results for all videos specified in
      `AnnotateVideoRequest`.
  """
    annotationResults = _messages.MessageField('GoogleCloudVideointelligenceV1beta2VideoAnnotationResults', 1, repeated=True)