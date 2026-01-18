from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesFetchInventoryRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesFetchInventoryRequest object.

  Fields:
    forceRefresh: If this flag is set to true, the source will be queried
      instead of using cached results. Using this flag will make the call
      slower.
    pageSize: The maximum number of VMs to return. The service may return
      fewer than this value. For AWS source: If unspecified, at most 500 VMs
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000. For VMWare source: If unspecified, all VMs will be
      returned. There is no limit for maximum value.
    pageToken: A page token, received from a previous `FetchInventory` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `FetchInventory` must match the call that
      provided the page token.
    source: Required. The name of the Source.
  """
    forceRefresh = _messages.BooleanField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    source = _messages.StringField(4, required=True)