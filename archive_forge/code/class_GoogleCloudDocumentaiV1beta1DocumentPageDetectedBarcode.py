from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentPageDetectedBarcode(_messages.Message):
    """A detected barcode.

  Fields:
    barcode: Detailed barcode information of the DetectedBarcode.
    layout: Layout for DetectedBarcode.
  """
    barcode = _messages.MessageField('GoogleCloudDocumentaiV1beta1Barcode', 1)
    layout = _messages.MessageField('GoogleCloudDocumentaiV1beta1DocumentPageLayout', 2)