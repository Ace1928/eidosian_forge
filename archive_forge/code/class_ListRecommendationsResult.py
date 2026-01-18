from decimal import Decimal
from boto.compat import filter, map
class ListRecommendationsResult(ResponseElement):
    ListingQualityRecommendations = MemberList(ItemIdentifier=Element())