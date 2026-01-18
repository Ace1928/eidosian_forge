from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3DocumentId(_messages.Message):
    """Document Identifier.

  Fields:
    gcsManagedDocId: A document id within user-managed Cloud Storage.
    revisionRef: Points to a specific revision of the document if set.
    unmanagedDocId: A document id within unmanaged dataset.
  """
    gcsManagedDocId = _messages.MessageField('GoogleCloudDocumentaiV1beta3DocumentIdGCSManagedDocumentId', 1)
    revisionRef = _messages.MessageField('GoogleCloudDocumentaiV1beta3RevisionRef', 2)
    unmanagedDocId = _messages.MessageField('GoogleCloudDocumentaiV1beta3DocumentIdUnmanagedDocumentId', 3)