from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3SampleDocumentsResponse(_messages.Message):
    """Response of the sample documents operation.

  Fields:
    sampleTestStatus: The status of sampling documents in test split.
    sampleTrainingStatus: The status of sampling documents in training split.
    selectedDocuments: The result of the sampling process.
  """
    sampleTestStatus = _messages.MessageField('GoogleRpcStatus', 1)
    sampleTrainingStatus = _messages.MessageField('GoogleRpcStatus', 2)
    selectedDocuments = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3SampleDocumentsResponseSelectedDocument', 3, repeated=True)