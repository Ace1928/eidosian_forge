from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaImageBoundingBoxAnnotation(_messages.Message):
    """Annotation details specific to image object detection.

  Fields:
    annotationSpecId: The resource Id of the AnnotationSpec that this
      Annotation pertains to.
    displayName: The display name of the AnnotationSpec that this Annotation
      pertains to.
    xMax: The rightmost coordinate of the bounding box.
    xMin: The leftmost coordinate of the bounding box.
    yMax: The bottommost coordinate of the bounding box.
    yMin: The topmost coordinate of the bounding box.
  """
    annotationSpecId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    xMax = _messages.FloatField(3)
    xMin = _messages.FloatField(4)
    yMax = _messages.FloatField(5)
    yMin = _messages.FloatField(6)