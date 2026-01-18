from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageAnchorPageRef(_messages.Message):
    """Represents a weak reference to a page element within a document.

  Enums:
    LayoutTypeValueValuesEnum: Optional. The type of the layout element that
      is being referenced if any.

  Fields:
    boundingPoly: Optional. Identifies the bounding polygon of a layout
      element on the page. If `layout_type` is set, the bounding polygon must
      be exactly the same to the layout element it's referring to.
    confidence: Optional. Confidence of detected page element, if applicable.
      Range `[0, 1]`.
    layoutId: Optional. Deprecated. Use PageRef.bounding_poly instead.
    layoutType: Optional. The type of the layout element that is being
      referenced if any.
    page: Required. Index into the Document.pages element, for example using
      `Document.pages` to locate the related page element. This field is
      skipped when its value is the default `0`. See
      https://developers.google.com/protocol-buffers/docs/proto3#json.
  """

    class LayoutTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the layout element that is being referenced if
    any.

    Values:
      LAYOUT_TYPE_UNSPECIFIED: Layout Unspecified.
      BLOCK: References a Page.blocks element.
      PARAGRAPH: References a Page.paragraphs element.
      LINE: References a Page.lines element.
      TOKEN: References a Page.tokens element.
      VISUAL_ELEMENT: References a Page.visual_elements element.
      TABLE: Refrrences a Page.tables element.
      FORM_FIELD: References a Page.form_fields element.
    """
        LAYOUT_TYPE_UNSPECIFIED = 0
        BLOCK = 1
        PARAGRAPH = 2
        LINE = 3
        TOKEN = 4
        VISUAL_ELEMENT = 5
        TABLE = 6
        FORM_FIELD = 7
    boundingPoly = _messages.MessageField('GoogleCloudDocumentaiV1beta1BoundingPoly', 1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    layoutId = _messages.StringField(3)
    layoutType = _messages.EnumField('LayoutTypeValueValuesEnum', 4)
    page = _messages.IntegerField(5)