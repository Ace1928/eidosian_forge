from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ListAzureClustersResponse(_messages.Message):
    """Response message for `AzureClusters.ListAzureClusters` method.

  Fields:
    azureClusters: A list of AzureCluster resources in the specified Google
      Cloud Platform project and region region.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    azureClusters = _messages.MessageField('GoogleCloudGkemulticloudV1AzureCluster', 1, repeated=True)
    nextPageToken = _messages.StringField(2)