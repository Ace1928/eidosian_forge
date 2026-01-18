from decimal import Decimal
from boto.compat import filter, map
class AuthorizationDetails(ResponseElement):
    AuthorizationAmount = Element(ComplexMoney)
    CapturedAmount = Element(ComplexMoney)
    AuthorizationFee = Element(ComplexMoney)
    AuthorizationStatus = Element()