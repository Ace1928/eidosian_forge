from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInventoriesResponse(_messages.Message):
    """A response message for listing inventory data for all VMs in a specified
  location.

  Fields:
    inventories: List of inventory objects.
    nextPageToken: The pagination token to retrieve the next page of inventory
      objects.
  """
    inventories = _messages.MessageField('Inventory', 1, repeated=True)
    nextPageToken = _messages.StringField(2)