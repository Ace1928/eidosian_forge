from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsPatchRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsPatchRequest
  object.

  Fields:
    googleCloudGkemulticloudV1AzureNodePool: A
      GoogleCloudGkemulticloudV1AzureNodePool resource to be passed as the
      request body.
    name: The name of this resource. Node pool names are formatted as
      `projects//locations//azureClusters//azureNodePools/`. For more details
      on Google Cloud resource names, see [Resource
      Names](https://cloud.google.com/apis/design/resource_names)
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field can
      only include these fields from AzureNodePool: *. `annotations`. *
      `version`. * `autoscaling.min_node_count`. *
      `autoscaling.max_node_count`. * `config.ssh_config.authorized_key`. *
      `management.auto_repair`. * `management`.
    validateOnly: If set, only validate the request, but don't actually update
      the node pool.
  """
    googleCloudGkemulticloudV1AzureNodePool = _messages.MessageField('GoogleCloudGkemulticloudV1AzureNodePool', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)