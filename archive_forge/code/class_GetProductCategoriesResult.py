from decimal import Decimal
from boto.compat import filter, map
class GetProductCategoriesResult(ResponseElement):
    Self = ElementList(ProductCategory)