from decimal import Decimal
from boto.compat import filter, map
class RefundResult(ResponseElement):
    RefundDetails = Element(RefundDetails)