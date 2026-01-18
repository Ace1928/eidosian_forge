from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UndeployIndexRequest(_messages.Message):
    """Request message for IndexEndpointService.UndeployIndex.

  Fields:
    deployedIndexId: Required. The ID of the DeployedIndex to be undeployed
      from the IndexEndpoint.
  """
    deployedIndexId = _messages.StringField(1)