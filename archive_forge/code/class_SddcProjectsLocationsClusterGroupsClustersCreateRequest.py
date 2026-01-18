from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsClustersCreateRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsClustersCreateRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    clusterId: Required. The user-provided ID of the `Cluster` to create. This
      ID must be unique among `Clusters` within the parent and becomes the
      final token in the name URI.
    managementCluster: Required. Deprecated. Use the management property in
      the `Cluster` resource. Whether the cluster is the management cluster.
    parent: Required. The `ClusterGroup` in where the new Cluster will be
      created. For example, projects/PROJECT-NUMBER/locations/us-
      central1/clusterGroups/ MY_GROUP
  """
    cluster = _messages.MessageField('Cluster', 1)
    clusterId = _messages.StringField(2)
    managementCluster = _messages.BooleanField(3)
    parent = _messages.StringField(4, required=True)