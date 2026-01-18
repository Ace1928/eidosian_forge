from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NearestNeighborSearchOperationMetadataContentValidationStats(_messages.Message):
    """A GoogleCloudAiplatformV1beta1NearestNeighborSearchOperationMetadataCont
  entValidationStats object.

  Fields:
    invalidRecordCount: Number of records in this file we skipped due to
      validate errors.
    partialErrors: The detail information of the partial failures encountered
      for those invalid records that couldn't be parsed. Up to 50 partial
      errors will be reported.
    sourceGcsUri: Cloud Storage URI pointing to the original file in user's
      bucket.
    validRecordCount: Number of records in this file that were successfully
      processed.
  """
    invalidRecordCount = _messages.IntegerField(1)
    partialErrors = _messages.MessageField('GoogleCloudAiplatformV1beta1NearestNeighborSearchOperationMetadataRecordError', 2, repeated=True)
    sourceGcsUri = _messages.StringField(3)
    validRecordCount = _messages.IntegerField(4)