from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsIpAddressesListRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsIpAddressesListRequest object.

  Fields:
    filter: List filter.
    pageSize: The maximum number of `IpAddress` objects to return. The service
      may return fewer than this value.
    pageToken: A page token, received from a previous `ListIpAddressesRequest`
      call. Provide this to retrieve the subsequent page. When paginating, you
      must provide exactly the same parameters to `ListIpAddressesRequest` as
      you provided to the page token request
    parent: Required. The parent ClusterGroup of which the IpAddresses belong
      to. For example: projects/PROJECT-NUMBER/locations/us-
      central1/clusterGroups/ MY-GROUP
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)