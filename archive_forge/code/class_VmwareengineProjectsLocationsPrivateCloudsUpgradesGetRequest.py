from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsUpgradesGetRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsUpgradesGetRequest object.

  Fields:
    name: Required. The name of the `Upgrade` resource to be retrieved.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-west1-a/privateClouds/my-
      cloud/upgrades/my-upgrade`
  """
    name = _messages.StringField(1, required=True)