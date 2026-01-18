from decimal import Decimal
from boto.compat import filter, map
class ListInboundShipmentItemsResult(ResponseElement):
    ItemData = MemberList()