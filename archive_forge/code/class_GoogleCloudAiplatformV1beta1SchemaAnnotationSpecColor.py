from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaAnnotationSpecColor(_messages.Message):
    """An entry of mapping between color and AnnotationSpec. The mapping is
  used in segmentation mask.

  Fields:
    color: The color of the AnnotationSpec in a segmentation mask.
    displayName: The display name of the AnnotationSpec represented by the
      color in the segmentation mask.
    id: The ID of the AnnotationSpec represented by the color in the
      segmentation mask.
  """
    color = _messages.MessageField('GoogleTypeColor', 1)
    displayName = _messages.StringField(2)
    id = _messages.StringField(3)