from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1TextAnnotation(_messages.Message):
    """TextAnnotation contains a structured representation of OCR extracted
  text. The hierarchy of an OCR extracted text structure is like this:
  TextAnnotation -> Page -> Block -> Paragraph -> Word -> Symbol Each
  structural component, starting from Page, may further have their own
  properties. Properties describe detected languages, breaks etc.. Please
  refer to the TextAnnotation.TextProperty message definition below for more
  detail.

  Fields:
    pages: List of pages detected by OCR.
    text: UTF-8 text detected on the pages.
  """
    pages = _messages.MessageField('GoogleCloudVisionV1p1beta1Page', 1, repeated=True)
    text = _messages.StringField(2)