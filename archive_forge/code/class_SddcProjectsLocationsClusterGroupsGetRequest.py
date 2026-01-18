from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsGetRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsGetRequest object.

  Fields:
    name: Required. The resource name of the `ClusterGroup` to retrieve.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-central1/clusterGroups/MY_GROUP
  """
    name = _messages.StringField(1, required=True)