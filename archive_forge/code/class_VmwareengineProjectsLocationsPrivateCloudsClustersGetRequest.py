from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsClustersGetRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsClustersGetRequest object.

  Fields:
    name: Required. The cluster resource name to retrieve. Resource names are
      schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/clusters/my-cluster`
  """
    name = _messages.StringField(1, required=True)