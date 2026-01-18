from decimal import Decimal
from boto.compat import filter, map
class OrderItem(ResponseElement):
    ItemPrice = Element(ComplexMoney)
    ShippingPrice = Element(ComplexMoney)
    GiftWrapPrice = Element(ComplexMoney)
    ItemTax = Element(ComplexMoney)
    ShippingTax = Element(ComplexMoney)
    GiftWrapTax = Element(ComplexMoney)
    ShippingDiscount = Element(ComplexMoney)
    PromotionDiscount = Element(ComplexMoney)
    PromotionIds = SimpleList()
    CODFee = Element(ComplexMoney)
    CODFeeDiscount = Element(ComplexMoney)