from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsGetRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsGetRequest
  object.

  Fields:
    name: Required. The name of the AzureNodePool resource to describe.
      `AzureNodePool` names are formatted as
      `projects//locations//azureClusters//azureNodePools/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    name = _messages.StringField(1, required=True)