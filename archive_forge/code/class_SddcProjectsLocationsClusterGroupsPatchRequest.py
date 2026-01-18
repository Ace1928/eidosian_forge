from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsPatchRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsPatchRequest object.

  Fields:
    clusterGroup: A ClusterGroup resource to be passed as the request body.
    name: Output only. The resource name of this `ClusterGroup`. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-central1/clusterGroups/MY-GROUP
    updateMask: Mask of fields to update. You must provide at least one path
      in this field. The elements of the repeated paths field may only include
      the following fields: "description" "labels"
      "network_config.external_ip_access"
  """
    clusterGroup = _messages.MessageField('ClusterGroup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)