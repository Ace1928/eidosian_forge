from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1BatchProcessMetadataIndividualProcessStatus(_messages.Message):
    """The status of a each individual document in the batch process.

  Fields:
    humanReviewStatus: The status of human review on the processed document.
    inputGcsSource: The source of the document, same as the input_gcs_source
      field in the request when the batch process started.
    outputGcsDestination: The Cloud Storage output destination (in the request
      as DocumentOutputConfig.GcsOutputConfig.gcs_uri) of the processed
      document if it was successful, otherwise empty.
    status: The status processing the document.
  """
    humanReviewStatus = _messages.MessageField('GoogleCloudDocumentaiV1HumanReviewStatus', 1)
    inputGcsSource = _messages.StringField(2)
    outputGcsDestination = _messages.StringField(3)
    status = _messages.MessageField('GoogleRpcStatus', 4)