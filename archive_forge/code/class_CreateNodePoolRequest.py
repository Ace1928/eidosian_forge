from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateNodePoolRequest(_messages.Message):
    """CreateNodePoolRequest creates a node pool for a cluster.

  Fields:
    clusterId: Deprecated. The name of the cluster. This field has been
      deprecated and replaced by the parent field.
    nodePool: Required. The node pool to create.
    parent: The parent (project, location, cluster name) where the node pool
      will be created. Specified in the format
      `projects/*/locations/*/clusters/*`.
    projectId: Deprecated. The Google Developers Console [project ID or
      project number](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects). This field has been deprecated and replaced by the
      parent field.
    zone: Deprecated. The name of the Google Compute Engine
      [zone](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster resides. This field has been deprecated and replaced by the
      parent field.
  """
    clusterId = _messages.StringField(1)
    nodePool = _messages.MessageField('NodePool', 2)
    parent = _messages.StringField(3)
    projectId = _messages.StringField(4)
    zone = _messages.StringField(5)