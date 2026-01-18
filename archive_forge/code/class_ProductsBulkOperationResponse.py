from decimal import Decimal
from boto.compat import filter, map
class ProductsBulkOperationResponse(ResponseResultList):
    _ResultClass = ProductsBulkOperationResult