from decimal import Decimal
from boto.compat import filter, map
class ListMarketplaceParticipationsResult(ResponseElement):
    ListParticipations = Element(Participation=ElementList())
    ListMarketplaces = Element(Marketplace=ElementList())