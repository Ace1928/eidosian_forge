from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupBackupsGetRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupBackupsGetRequest object.

  Fields:
    name: Required. The resource name of the `ClusterGroupBackup` to retrieve.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-
      central1/clusterGroupBackups/MY_BACKUP
  """
    name = _messages.StringField(1, required=True)