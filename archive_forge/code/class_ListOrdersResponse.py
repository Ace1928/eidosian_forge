from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOrdersResponse(_messages.Message):
    """Response message for the ListOrders method.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    orders: The list of Order. If the `{location}` value in the request is
      "-", the response contains a list of instances from all locations. In
      case any location is unreachable, the response will only return
      management servers in reachable locations and the 'unreachable' field
      will be populated with a list of unreachable locations.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    orders = _messages.MessageField('Order', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)