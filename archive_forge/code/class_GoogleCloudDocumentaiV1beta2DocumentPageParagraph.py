from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageParagraph(_messages.Message):
    """A collection of lines that a human would perceive as a paragraph.

  Fields:
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for Paragraph.
    provenance: The history of this annotation.
  """
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageDetectedLanguage', 1, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageLayout', 2)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentProvenance', 3)