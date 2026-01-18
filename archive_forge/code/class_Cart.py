from decimal import Decimal
from boto.compat import filter, map
class Cart(ResponseElement):
    ActiveCartItemList = Element(CartItem=ElementList(CartItem))
    SavedCartItemList = Element(CartItem=ElementList(CartItem))