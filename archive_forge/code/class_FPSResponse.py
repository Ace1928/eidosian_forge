from decimal import Decimal
from boto.compat import filter, map
class FPSResponse(Response):
    _action = action
    _Result = globals().get(action + 'Result', ResponseElement)

    def endElement(self, name, value, connection):
        if name != action + 'Response':
            super(FPSResponse, self).endElement(name, value, connection)