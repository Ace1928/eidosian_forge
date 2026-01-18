from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageSymbol(_messages.Message):
    """A detected symbol.

  Fields:
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for Symbol.
  """
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageDetectedLanguage', 1, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageLayout', 2)