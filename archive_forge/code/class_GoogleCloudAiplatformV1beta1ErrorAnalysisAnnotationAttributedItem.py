from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ErrorAnalysisAnnotationAttributedItem(_messages.Message):
    """Attributed items for a given annotation, typically representing
  neighbors from the training sets constrained by the query type.

  Fields:
    annotationResourceName: The unique ID for each annotation. Used by FE to
      allocate the annotation in DB.
    distance: The distance of this item to the annotation.
  """
    annotationResourceName = _messages.StringField(1)
    distance = _messages.FloatField(2)