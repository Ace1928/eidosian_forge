from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsDeleteRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersAzureNodePoolsDeleteRequest
  object.

  Fields:
    allowMissing: If set to true, and the AzureNodePool resource is not found,
      the request will succeed but no action will be taken on the server and a
      completed Operation will be returned. Useful for idempotent deletion.
    etag: The current ETag of the AzureNodePool. Allows clients to perform
      deletions through optimistic concurrency control. If the provided ETag
      does not match the current etag of the node pool, the request will fail
      and an ABORTED error will be returned.
    ignoreErrors: Optional. If set to true, the deletion of AzureNodePool
      resource will succeed even if errors occur during deleting in node pool
      resources. Using this parameter may result in orphaned resources in the
      node pool.
    name: Required. The resource name the AzureNodePool to delete.
      `AzureNodePool` names are formatted as
      `projects//locations//azureClusters//azureNodePools/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      delete the node pool.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    ignoreErrors = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    validateOnly = _messages.BooleanField(5)