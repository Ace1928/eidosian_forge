from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentTextChange(_messages.Message):
    """This message is used for text changes aka. OCR corrections.

  Fields:
    changedText: The text that replaces the text identified in the
      `text_anchor`.
    provenance: The history of this annotation.
    textAnchor: Provenance of the correction. Text anchor indexing into the
      Document.text. There can only be a single `TextAnchor.text_segments`
      element. If the start and end index of the text segment are the same,
      the text change is inserted before that index.
  """
    changedText = _messages.StringField(1)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentProvenance', 2, repeated=True)
    textAnchor = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentTextAnchor', 3)