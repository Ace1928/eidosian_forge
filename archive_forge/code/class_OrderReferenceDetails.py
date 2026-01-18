from decimal import Decimal
from boto.compat import filter, map
class OrderReferenceDetails(ResponseElement):
    Buyer = Element()
    OrderTotal = Element(ComplexMoney)
    Destination = Element(PhysicalDestination=Element())
    SellerOrderAttributes = Element()
    OrderReferenceStatus = Element()
    Constraints = ElementList()