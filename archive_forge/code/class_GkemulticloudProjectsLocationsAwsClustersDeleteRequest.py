from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersDeleteRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the AwsCluster resource is not found,
      the request will succeed but no action will be taken on the server and a
      completed Operation will be returned. Useful for idempotent deletion.
    etag: The current etag of the AwsCluster. Allows clients to perform
      deletions through optimistic concurrency control. If the provided etag
      does not match the current etag of the cluster, the request will fail
      and an ABORTED error will be returned.
    ignoreErrors: Optional. If set to true, the deletion of AwsCluster
      resource will succeed even if errors occur during deleting in cluster
      resources. Using this parameter may result in orphaned resources in the
      cluster.
    name: Required. The resource name the AwsCluster to delete. `AwsCluster`
      names are formatted as `projects//locations//awsClusters/`. See
      [Resource Names](https://cloud.google.com/apis/design/resource_names)
      for more details on Google Cloud Platform resource names.
    validateOnly: If set, only validate the request, but do not actually
      delete the resource.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    ignoreErrors = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    validateOnly = _messages.BooleanField(5)