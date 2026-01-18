from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageTable(_messages.Message):
    """A table representation similar to HTML table structure.

  Fields:
    bodyRows: Body rows of the table.
    detectedLanguages: A list of detected languages together with confidence.
    headerRows: Header rows of the table.
    layout: Layout for Table.
    provenance: The history of this table.
  """
    bodyRows = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageTableTableRow', 1, repeated=True)
    detectedLanguages = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageDetectedLanguage', 2, repeated=True)
    headerRows = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageTableTableRow', 3, repeated=True)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageLayout', 4)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentProvenance', 5)