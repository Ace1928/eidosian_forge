from decimal import Decimal
from boto.compat import filter, map
class GetAccountActivityResult(ResponseElement):

    def __init__(self, *args, **kw):
        self.Transaction = []
        super(GetAccountActivityResult, self).__init__(*args, **kw)

    def startElement(self, name, attrs, connection):
        if name == 'Transaction':
            getattr(self, name).append(Transaction(name=name))
            return getattr(self, name)[-1]
        return super(GetAccountActivityResult, self).startElement(name, attrs, connection)