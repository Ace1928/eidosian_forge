from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsIpAddressesDeleteRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsIpAddressesDeleteRequest object.

  Fields:
    name: Required. The resource name of the `ClusterGroupBackup` to be
      deleted.
  """
    name = _messages.StringField(1, required=True)