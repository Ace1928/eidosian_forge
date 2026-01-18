from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersDeleteRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalClustersDeleteRequest object.

  Fields:
    allowMissing: If set to true, and the bare metal cluster is not found, the
      request will succeed but no action will be taken on the server and
      return a completed LRO.
    etag: The current etag of the bare metal Cluster. If an etag is provided
      and does not match the current etag of the cluster, deletion will be
      blocked and an ABORTED error will be returned.
    force: If set to true, any node pools from the cluster will also be
      deleted.
    ignoreErrors: If set to true, the deletion of a bare metal user cluster
      resource will succeed even if errors occur during deletion. This
      parameter can be used when you want to delete GCP's cluster resource and
      the on-prem admin cluster that hosts your user cluster is disconnected /
      unreachable or deleted. WARNING: Using this parameter when your user
      cluster still exists may result in a deleted GCP user cluster but an
      existing on-prem user cluster.
    name: Required. Name of the bare metal user cluster to be deleted. Format:
      "projects/{project}/locations/{location}/bareMetalClusters/{bare_metal_c
      luster}"
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    force = _messages.BooleanField(3)
    ignoreErrors = _messages.BooleanField(4)
    name = _messages.StringField(5, required=True)
    validateOnly = _messages.BooleanField(6)