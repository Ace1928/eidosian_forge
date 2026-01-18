from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTextExtractionAnnotation(_messages.Message):
    """Annotation details specific to text extraction.

  Fields:
    annotationSpecId: The resource Id of the AnnotationSpec that this
      Annotation pertains to.
    displayName: The display name of the AnnotationSpec that this Annotation
      pertains to.
    textSegment: The segment of the text content.
  """
    annotationSpecId = _messages.StringField(1)
    displayName = _messages.StringField(2)
    textSegment = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTextSegment', 3)