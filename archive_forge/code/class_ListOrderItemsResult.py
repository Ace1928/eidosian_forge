from decimal import Decimal
from boto.compat import filter, map
class ListOrderItemsResult(ResponseElement):
    OrderItems = Element(OrderItem=ElementList(OrderItem))