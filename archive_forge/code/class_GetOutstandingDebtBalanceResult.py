from decimal import Decimal
from boto.compat import filter, map
class GetOutstandingDebtBalanceResult(ResponseElement):

    def startElement(self, name, attrs, connection):
        if name == 'OutstandingDebt':
            setattr(self, name, AmountCollection(name=name))
            return getattr(self, name)
        return super(GetOutstandingDebtBalanceResult, self).startElement(name, attrs, connection)