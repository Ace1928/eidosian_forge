from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerProjectsZonesClustersNodePoolsDeleteRequest(_messages.Message):
    """A ContainerProjectsZonesClustersNodePoolsDeleteRequest object.

  Fields:
    clusterId: Deprecated. The name of the cluster. This field has been
      deprecated and replaced by the name field.
    name: The name (project, location, cluster, node pool id) of the node pool
      to delete. Specified in the format
      `projects/*/locations/*/clusters/*/nodePools/*`.
    nodePoolId: Deprecated. The name of the node pool to delete. This field
      has been deprecated and replaced by the name field.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      name field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      name field.
  """
    clusterId = _messages.StringField(1, required=True)
    name = _messages.StringField(2)
    nodePoolId = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4, required=True)
    zone = _messages.StringField(5, required=True)