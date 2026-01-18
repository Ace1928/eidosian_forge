from decimal import Decimal
from boto.compat import filter, map
class CreateInboundShipmentPlanResult(ResponseElement):
    InboundShipmentPlans = MemberList(ShipToAddress=Element(), Items=MemberList())