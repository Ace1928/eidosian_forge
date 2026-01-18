import xml.sax
import cgi
from boto.compat import six, StringIO
class ItemSet(ResponseGroup):
    """A special ResponseGroup that has built-in paging, and
    only creates new Items on the "Item" tag"""

    def __init__(self, connection, action, params, page=0):
        ResponseGroup.__init__(self, connection, 'Items')
        self.objs = []
        self.iter = None
        self.page = page
        self.action = action
        self.params = params
        self.curItem = None
        self.total_results = 0
        self.total_pages = 0
        self.is_valid = False
        self.errors = []

    def startElement(self, name, attrs, connection):
        if name == 'Item':
            self.curItem = Item(self._connection)
        elif self.curItem is not None:
            self.curItem.startElement(name, attrs, connection)
        return None

    def endElement(self, name, value, connection):
        if name == 'TotalResults':
            self.total_results = value
        elif name == 'TotalPages':
            self.total_pages = value
        elif name == 'IsValid':
            if value == 'True':
                self.is_valid = True
        elif name == 'Code':
            self.errors.append({'Code': value, 'Message': None})
        elif name == 'Message':
            self.errors[-1]['Message'] = value
        elif name == 'Item':
            self.objs.append(self.curItem)
            self._xml.write(self.curItem.to_xml())
            self.curItem = None
        elif self.curItem is not None:
            self.curItem.endElement(name, value, connection)
        return None

    def __next__(self):
        """Special paging functionality"""
        if self.iter is None:
            self.iter = iter(self.objs)
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = None
            self.objs = []
            if int(self.page) < int(self.total_pages):
                self.page += 1
                self._connection.get_response(self.action, self.params, self.page, self)
                return next(self)
            else:
                raise
    next = __next__

    def __iter__(self):
        return self

    def to_xml(self):
        """Override to first fetch everything"""
        for item in self:
            pass
        return ResponseGroup.to_xml(self)