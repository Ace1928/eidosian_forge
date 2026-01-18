from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1MutateDeployedIndexResponse(_messages.Message):
    """Response message for IndexEndpointService.MutateDeployedIndex.

  Fields:
    deployedIndex: The DeployedIndex that had been updated in the
      IndexEndpoint.
  """
    deployedIndex = _messages.MessageField('GoogleCloudAiplatformV1DeployedIndex', 1)