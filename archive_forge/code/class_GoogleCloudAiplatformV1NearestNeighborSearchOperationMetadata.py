from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NearestNeighborSearchOperationMetadata(_messages.Message):
    """Runtime operation metadata with regard to Matching Engine Index.

  Fields:
    contentValidationStats: The validation stats of the content (per file) to
      be inserted or updated on the Matching Engine Index resource. Populated
      if contentsDeltaUri is provided as part of Index.metadata. Please note
      that, currently for those files that are broken or has unsupported file
      format, we will not have the stats for those files.
    dataBytesCount: The ingested data size in bytes.
  """
    contentValidationStats = _messages.MessageField('GoogleCloudAiplatformV1NearestNeighborSearchOperationMetadataContentValidationStats', 1, repeated=True)
    dataBytesCount = _messages.IntegerField(2)