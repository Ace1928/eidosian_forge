from decimal import Decimal
from boto.compat import filter, map
class SimpleList(DeclarativeType):

    def __init__(self, *args, **kw):
        super(SimpleList, self).__init__(*args, **kw)
        self._value = []

    def start(self, *args, **kw):
        return None

    def end(self, name, value, *args, **kw):
        self._value.append(value)