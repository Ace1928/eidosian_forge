from decimal import Decimal
from boto.compat import filter, map
class GetSubscriptionResult(ResponseElement):
    Subscription = Element(Subscription)