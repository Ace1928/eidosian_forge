from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeployIndexRequest(_messages.Message):
    """Request message for IndexEndpointService.DeployIndex.

  Fields:
    deployedIndex: Required. The DeployedIndex to be created within the
      IndexEndpoint.
  """
    deployedIndex = _messages.MessageField('GoogleCloudAiplatformV1beta1DeployedIndex', 1)