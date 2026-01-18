from decimal import Decimal
from boto.compat import filter, map
class AuthorizeResult(ResponseElement):
    AuthorizationDetails = Element(AuthorizationDetails)