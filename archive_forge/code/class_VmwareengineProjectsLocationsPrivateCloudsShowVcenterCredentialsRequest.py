from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsShowVcenterCredentialsRequest(_messages.Message):
    """A
  VmwareengineProjectsLocationsPrivateCloudsShowVcenterCredentialsRequest
  object.

  Fields:
    privateCloud: Required. The resource name of the private cloud to be
      queried for credentials. Resource names are schemeless URIs that follow
      the conventions in https://cloud.google.com/apis/design/resource_names.
      For example: `projects/my-project/locations/us-
      central1-a/privateClouds/my-cloud`
    username: Optional. The username of the user to be queried for
      credentials. The default value of this field is CloudOwner@gve.local.
      The provided value must be one of the following: CloudOwner@gve.local,
      solution-user-01@gve.local, solution-user-02@gve.local, solution-
      user-03@gve.local, solution-user-04@gve.local, solution-
      user-05@gve.local, zertoadmin@gve.local.
  """
    privateCloud = _messages.StringField(1, required=True)
    username = _messages.StringField(2)