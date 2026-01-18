from decimal import Decimal
from boto.compat import filter, map
class GetPackageTrackingDetailsResult(ResponseElement):
    ShipToAddress = Element()
    TrackingEvents = MemberList(EventAddress=Element())