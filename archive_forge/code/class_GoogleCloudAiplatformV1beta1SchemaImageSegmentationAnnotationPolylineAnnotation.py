from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaImageSegmentationAnnotationPolylineAnnotation(_messages.Message):
    """Represents a polyline in image.

  Fields:
    annotationSpecId: The resource Id of the AnnotationSpec that this
      Annotation pertains to.
    displayName: The display name of the AnnotationSpec that this Annotation
      pertains to.
    vertexes: The vertexes are connected one by one and the last vertex in not
      connected to the first one.
  """
    annotationSpecId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    vertexes = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaVertex', 3, repeated=True)