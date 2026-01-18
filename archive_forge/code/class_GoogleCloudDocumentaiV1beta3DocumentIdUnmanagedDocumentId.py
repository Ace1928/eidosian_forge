from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3DocumentIdUnmanagedDocumentId(_messages.Message):
    """Identifies a document uniquely within the scope of a dataset in
  unmanaged option.

  Fields:
    docId: Required. The id of the document.
  """
    docId = _messages.StringField(1)