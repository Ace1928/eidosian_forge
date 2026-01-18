from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1UndeployIndexOperationMetadata(_messages.Message):
    """Runtime operation information for IndexEndpointService.UndeployIndex.

  Fields:
    genericMetadata: The operation generic information.
  """
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 1)