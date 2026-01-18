from decimal import Decimal
from boto.compat import filter, map
class CaptureDetails(ResponseElement):
    CaptureAmount = Element(ComplexMoney)
    RefundedAmount = Element(ComplexMoney)
    CaptureFee = Element(ComplexMoney)
    CaptureStatus = Element()