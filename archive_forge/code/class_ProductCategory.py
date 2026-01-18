from decimal import Decimal
from boto.compat import filter, map
class ProductCategory(ResponseElement):

    def __init__(self, *args, **kw):
        setattr(self, 'Parent', Element(ProductCategory))
        super(ProductCategory, self).__init__(*args, **kw)