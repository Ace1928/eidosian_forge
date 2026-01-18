from decimal import Decimal
from boto.compat import filter, map
class GetTokensResult(ResponseElement):

    def __init__(self, *args, **kw):
        self.Token = []
        super(GetTokensResult, self).__init__(*args, **kw)

    def startElement(self, name, attrs, connection):
        if name == 'Token':
            getattr(self, name).append(ResponseElement(name=name))
            return getattr(self, name)[-1]
        return super(GetTokensResult, self).startElement(name, attrs, connection)