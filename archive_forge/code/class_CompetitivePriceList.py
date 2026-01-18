from decimal import Decimal
from boto.compat import filter, map
class CompetitivePriceList(ResponseElement):
    CompetitivePrice = ElementList(CompetitivePrice)