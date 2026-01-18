from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsHumanReviewConfigReviewDocumentRequest(_messages.Message):
    """A
  DocumentaiProjectsLocationsProcessorsHumanReviewConfigReviewDocumentRequest
  object.

  Fields:
    googleCloudDocumentaiV1ReviewDocumentRequest: A
      GoogleCloudDocumentaiV1ReviewDocumentRequest resource to be passed as
      the request body.
    humanReviewConfig: Required. The resource name of the HumanReviewConfig
      that the document will be reviewed with.
  """
    googleCloudDocumentaiV1ReviewDocumentRequest = _messages.MessageField('GoogleCloudDocumentaiV1ReviewDocumentRequest', 1)
    humanReviewConfig = _messages.StringField(2, required=True)