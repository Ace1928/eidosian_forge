from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaImageSegmentationAnnotation(_messages.Message):
    """Annotation details specific to image segmentation.

  Fields:
    maskAnnotation: Mask based segmentation annotation. Only one mask
      annotation can exist for one image.
    polygonAnnotation: Polygon annotation.
    polylineAnnotation: Polyline annotation.
  """
    maskAnnotation = _messages.MessageField('GoogleCloudAiplatformV1SchemaImageSegmentationAnnotationMaskAnnotation', 1)
    polygonAnnotation = _messages.MessageField('GoogleCloudAiplatformV1SchemaImageSegmentationAnnotationPolygonAnnotation', 2)
    polylineAnnotation = _messages.MessageField('GoogleCloudAiplatformV1SchemaImageSegmentationAnnotationPolylineAnnotation', 3)