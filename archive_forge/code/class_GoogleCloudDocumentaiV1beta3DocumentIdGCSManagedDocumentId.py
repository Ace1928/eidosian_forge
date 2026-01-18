from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3DocumentIdGCSManagedDocumentId(_messages.Message):
    """Identifies a document uniquely within the scope of a dataset in the
  user-managed Cloud Storage option.

  Fields:
    cwDocId: Id of the document (indexed) managed by Content Warehouse.
    gcsUri: Required. The Cloud Storage URI where the actual document is
      stored.
  """
    cwDocId = _messages.StringField(1)
    gcsUri = _messages.StringField(2)