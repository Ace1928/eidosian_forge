from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ListAzureNodePoolsResponse(_messages.Message):
    """Response message for `AzureClusters.ListAzureNodePools` method.

  Fields:
    azureNodePools: A list of AzureNodePool resources in the specified
      `AzureCluster`.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    azureNodePools = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodePool', 1, repeated=True)
    nextPageToken = _messages.StringField(2)