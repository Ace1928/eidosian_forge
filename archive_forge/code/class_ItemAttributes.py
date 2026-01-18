from decimal import Decimal
from boto.compat import filter, map
class ItemAttributes(AttributeSet):
    Languages = Element(Language=ElementList())

    def __init__(self, *args, **kw):
        names = ('Actor', 'Artist', 'Author', 'Creator', 'Director', 'Feature', 'Format', 'GemType', 'MaterialType', 'MediaType', 'OperatingSystem', 'Platform')
        for name in names:
            setattr(self, name, SimpleList())
        super(ItemAttributes, self).__init__(*args, **kw)