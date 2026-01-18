from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsIpAddressesCreateRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsIpAddressesCreateRequest object.

  Fields:
    ipAddress: A IpAddress resource to be passed as the request body.
    ipAddressId: Required. The user-provided ID of the `IpAddress` to create.
      This ID must be unique among `IpAddress` within the parent and becomes
      the final token in the name URI.
    parent: Required. The ClusterGroup in which the `IpAddress` will be
      created. For example, projects/PROJECT-NUMBER/locations/us-
      central1/clusterGroups/ MY-GROUP
  """
    ipAddress = _messages.MessageField('IpAddress', 1)
    ipAddressId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)