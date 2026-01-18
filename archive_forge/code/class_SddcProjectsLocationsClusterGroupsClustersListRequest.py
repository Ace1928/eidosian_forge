from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsClustersListRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsClustersListRequest object.

  Fields:
    filter: List filter.
    pageSize: The maximum number of clusters to return. The service might
      return fewer clusters.
    pageToken: A page token received from a previous `ListClustersRequest`.
      Provide this to retrieve the subsequent page. When paginating, you must
      provide exactly the same parameters to `ListClustersRequest` as you
      provided to the page token request.
    parent: Required. The project, location, and cluster group that is queried
      for clusters. For example, projects/PROJECT-NUMBER /locations/us-
      central1/clusterGroups/MY_GROUP
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)