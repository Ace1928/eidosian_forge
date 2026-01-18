from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionImageObjectDetectionPredictionResult(_messages.Message):
    """Prediction output format for Image Object Detection.

  Messages:
    BboxesValueListEntry: Single entry in a BboxesValue.

  Fields:
    bboxes: Bounding boxes, i.e. the rectangles over the image, that pinpoint
      the found AnnotationSpecs. Given in order that matches the IDs. Each
      bounding box is an array of 4 numbers `xMin`, `xMax`, `yMin`, and
      `yMax`, which represent the extremal coordinates of the box. They are
      relative to the image size, and the point 0,0 is in the top left of the
      image.
    confidences: The Model's confidences in correctness of the predicted IDs,
      higher value means higher confidence. Order matches the Ids.
    displayNames: The display names of the AnnotationSpecs that had been
      identified, order matches the IDs.
    ids: The resource IDs of the AnnotationSpecs that had been identified,
      ordered by the confidence score descendingly.
  """

    class BboxesValueListEntry(_messages.Message):
        """Single entry in a BboxesValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    bboxes = _messages.MessageField('BboxesValueListEntry', 1, repeated=True)
    confidences = _messages.FloatField(2, repeated=True, variant=_messages.Variant.FLOAT)
    displayNames = _messages.StringField(3, repeated=True)
    ids = _messages.IntegerField(4, repeated=True)