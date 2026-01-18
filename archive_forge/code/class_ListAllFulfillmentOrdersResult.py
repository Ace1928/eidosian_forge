from decimal import Decimal
from boto.compat import filter, map
class ListAllFulfillmentOrdersResult(ResponseElement):
    FulfillmentOrders = MemberList(FulfillmentOrder)