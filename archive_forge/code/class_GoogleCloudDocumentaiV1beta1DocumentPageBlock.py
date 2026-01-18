from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageBlock(_messages.Message):
    """A block has a set of lines (collected into paragraphs) that have a
  common line-spacing and orientation.

  Fields:
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for Block.
    provenance: The history of this annotation.
  """
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageDetectedLanguage', 1, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageLayout', 2)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentProvenance', 3)