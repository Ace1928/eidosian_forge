from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsAddressGroupsCloneItemsRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsAddressGroupsCloneItemsRequest object.

  Fields:
    addressGroup: Required. A name of the AddressGroup to clone items to. Must
      be in the format
      `projects|organization/*/locations/{location}/addressGroups/*`.
    cloneAddressGroupItemsRequest: A CloneAddressGroupItemsRequest resource to
      be passed as the request body.
  """
    addressGroup = _messages.StringField(1, required=True)
    cloneAddressGroupItemsRequest = _messages.MessageField('CloneAddressGroupItemsRequest', 2)