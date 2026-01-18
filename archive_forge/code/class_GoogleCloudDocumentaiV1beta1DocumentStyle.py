from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentStyle(_messages.Message):
    """Annotation for common text style attributes. This adheres to CSS
  conventions as much as possible.

  Fields:
    backgroundColor: Text background color.
    color: Text color.
    fontFamily: Font family such as `Arial`, `Times New Roman`.
      https://www.w3schools.com/cssref/pr_font_font-family.asp
    fontSize: Font size.
    fontWeight: [Font
      weight](https://www.w3schools.com/cssref/pr_font_weight.asp). Possible
      values are `normal`, `bold`, `bolder`, and `lighter`.
    textAnchor: Text anchor indexing into the Document.text.
    textDecoration: [Text
      decoration](https://www.w3schools.com/cssref/pr_text_text-
      decoration.asp). Follows CSS standard.
    textStyle: [Text style](https://www.w3schools.com/cssref/pr_font_font-
      style.asp). Possible values are `normal`, `italic`, and `oblique`.
  """
    backgroundColor = _messages.MessageField('GoogleTypeColor', 1)
    color = _messages.MessageField('GoogleTypeColor', 2)
    fontFamily = _messages.StringField(3)
    fontSize = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentStyleFontSize', 4)
    fontWeight = _messages.StringField(5)
    textAnchor = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentTextAnchor', 6)
    textDecoration = _messages.StringField(7)
    textStyle = _messages.StringField(8)