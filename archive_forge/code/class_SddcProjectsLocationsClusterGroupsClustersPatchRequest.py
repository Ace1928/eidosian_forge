from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsClustersPatchRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsClustersPatchRequest object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    name: Output only. The resource name of this `Cluster`. Resource names are
      schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/REGION/clusterGroups/MY-
      GROUP/clusters/MY-CLUSTER
    updateMask: Mask of fields to update. You must provide at least one path
      in this field. The elements of the repeated paths field may only include
      these fields: "labels"
  """
    cluster = _messages.MessageField('Cluster', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)