from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2BatchProcessDocumentsResponse(_messages.Message):
    """Response to an batch document processing request. This is returned in
  the LRO Operation after the operation is complete.

  Fields:
    responses: Responses for each individual document.
  """
    responses = _messages.MessageField('GoogleCloudDocumentaiV1beta2ProcessDocumentResponse', 1, repeated=True)