from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageLine(_messages.Message):
    """A collection of tokens that a human would perceive as a line. Does not
  cross column boundaries, can be horizontal, vertical, etc.

  Fields:
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for Line.
    provenance: The history of this annotation.
  """
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageDetectedLanguage', 1, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1DocumentPageLayout', 2)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1DocumentProvenance', 3)