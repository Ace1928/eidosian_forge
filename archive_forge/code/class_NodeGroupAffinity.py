from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupAffinity(_messages.Message):
    """Node Group Affinity for clusters using sole-tenant node groups. The
  Dataproc NodeGroupAffinity resource is not related to the Dataproc NodeGroup
  resource.

  Fields:
    nodeGroupUri: Required. The URI of a sole-tenant node group resource
      (https://cloud.google.com/compute/docs/reference/rest/v1/nodeGroups)
      that the cluster will be created on.A full URL, partial URI, or node
      group name are valid. Examples: https://www.googleapis.com/compute/v1/pr
      ojects/[project_id]/zones/[zone]/nodeGroups/node-group-1
      projects/[project_id]/zones/[zone]/nodeGroups/node-group-1 node-group-1
  """
    nodeGroupUri = _messages.StringField(1)