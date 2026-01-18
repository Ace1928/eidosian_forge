from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1Block(_messages.Message):
    """Logical element on the page.

  Enums:
    BlockTypeValueValuesEnum: Detected block type (text, image etc) for this
      block.

  Fields:
    blockType: Detected block type (text, image etc) for this block.
    boundingBox: The bounding box for the block. The vertices are in the order
      of top-left, top-right, bottom-right, bottom-left. When a rotation of
      the bounding box is detected the rotation is represented as around the
      top-left corner as defined when the text is read in the 'natural'
      orientation. For example: * when the text is horizontal it might look
      like: 0----1 | | 3----2 * when it's rotated 180 degrees around the top-
      left corner it becomes: 2----3 | | 1----0 and the vertex order will
      still be (0, 1, 2, 3).
    confidence: Confidence of the OCR results on the block. Range [0, 1].
    paragraphs: List of paragraphs in this block (if this blocks is of type
      text).
    property: Additional information detected for the block.
  """

    class BlockTypeValueValuesEnum(_messages.Enum):
        """Detected block type (text, image etc) for this block.

    Values:
      UNKNOWN: Unknown block type.
      TEXT: Regular text block.
      TABLE: Table block.
      PICTURE: Image block.
      RULER: Horizontal/vertical line box.
      BARCODE: Barcode block.
    """
        UNKNOWN = 0
        TEXT = 1
        TABLE = 2
        PICTURE = 3
        RULER = 4
        BARCODE = 5
    blockType = _messages.EnumField('BlockTypeValueValuesEnum', 1)
    boundingBox = _messages.MessageField('GoogleCloudVisionV1p1beta1BoundingPoly', 2)
    confidence = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    paragraphs = _messages.MessageField('GoogleCloudVisionV1p1beta1Paragraph', 4, repeated=True)
    property = _messages.MessageField('GoogleCloudVisionV1p1beta1TextAnnotationTextProperty', 5)