from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsClustersNodesGetRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsClustersNodesGetRequest
  object.

  Fields:
    name: Required. The resource name of the node to retrieve. For example: `p
      rojects/{project}/locations/{location}/privateClouds/{private_cloud}/clu
      sters/{cluster}/nodes/{node}`
  """
    name = _messages.StringField(1, required=True)