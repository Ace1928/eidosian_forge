from decimal import Decimal
from boto.compat import filter, map
class ListOrdersResult(ResponseElement):
    Orders = Element(Order=ElementList(Order))