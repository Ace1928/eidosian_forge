from decimal import Decimal
from boto.compat import filter, map
class FulfillmentOrder(ResponseElement):
    DestinationAddress = Element()
    NotificationEmailList = MemberList(SimpleList)