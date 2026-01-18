from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageToken(_messages.Message):
    """A detected token.

  Fields:
    detectedBreak: Detected break at the end of a Token.
    detectedLanguages: A list of detected languages together with confidence.
    layout: Layout for Token.
    provenance: The history of this annotation.
    styleInfo: Text style attributes.
  """
    detectedBreak = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageTokenDetectedBreak', 1)
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageDetectedLanguage', 2, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageLayout', 3)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentProvenance', 4)
    styleInfo = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageTokenStyleInfo', 5)