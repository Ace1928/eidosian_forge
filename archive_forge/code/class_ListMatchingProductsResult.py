from decimal import Decimal
from boto.compat import filter, map
class ListMatchingProductsResult(ResponseElement):
    Products = Element(Product=ElementList(Product))