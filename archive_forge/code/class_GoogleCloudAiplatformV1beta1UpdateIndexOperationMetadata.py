from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UpdateIndexOperationMetadata(_messages.Message):
    """Runtime operation information for IndexService.UpdateIndex.

  Fields:
    genericMetadata: The operation generic information.
    nearestNeighborSearchOperationMetadata: The operation metadata with regard
      to Matching Engine Index operation.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenericOperationMetadata', 1)
    nearestNeighborSearchOperationMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1NearestNeighborSearchOperationMetadata', 2)