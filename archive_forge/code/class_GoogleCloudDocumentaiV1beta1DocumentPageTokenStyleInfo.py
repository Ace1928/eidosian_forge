from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageTokenStyleInfo(_messages.Message):
    """Font and other text style attributes.

  Fields:
    backgroundColor: Color of the background.
    bold: Whether the text is bold (equivalent to font_weight is at least
      `700`).
    fontSize: Font size in points (`1` point is `\\xb9\\u2044\\u2087\\u2082`
      inches).
    fontType: Name or style of the font.
    fontWeight: TrueType weight on a scale `100` (thin) to `1000` (ultra-
      heavy). Normal is `400`, bold is `700`.
    handwritten: Whether the text is handwritten.
    italic: Whether the text is italic.
    letterSpacing: Letter spacing in points.
    pixelFontSize: Font size in pixels, equal to _unrounded font_size_ *
      _resolution_ \\xf7 `72.0`.
    smallcaps: Whether the text is in small caps.
    strikeout: Whether the text is strikethrough.
    subscript: Whether the text is a subscript.
    superscript: Whether the text is a superscript.
    textColor: Color of the text.
    underlined: Whether the text is underlined.
  """
    backgroundColor = _messages.MessageField('GoogleTypeColor', 1)
    bold = _messages.BooleanField(2)
    fontSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    fontType = _messages.StringField(4)
    fontWeight = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    handwritten = _messages.BooleanField(6)
    italic = _messages.BooleanField(7)
    letterSpacing = _messages.FloatField(8)
    pixelFontSize = _messages.FloatField(9)
    smallcaps = _messages.BooleanField(10)
    strikeout = _messages.BooleanField(11)
    subscript = _messages.BooleanField(12)
    superscript = _messages.BooleanField(13)
    textColor = _messages.MessageField('GoogleTypeColor', 14)
    underlined = _messages.BooleanField(15)