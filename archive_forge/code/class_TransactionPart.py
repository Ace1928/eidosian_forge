from decimal import Decimal
from boto.compat import filter, map
class TransactionPart(ResponseElement):

    def startElement(self, name, attrs, connection):
        if name == 'FeesPaid':
            setattr(self, name, ComplexAmount(name=name))
            return getattr(self, name)
        return super(TransactionPart, self).startElement(name, attrs, connection)