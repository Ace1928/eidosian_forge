import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class CharacterData(Childless, Node):
    __slots__ = ('_data', 'ownerDocument', 'parentNode', 'previousSibling', 'nextSibling')

    def __init__(self):
        self.ownerDocument = self.parentNode = None
        self.previousSibling = self.nextSibling = None
        self._data = ''
        Node.__init__(self)

    def _get_length(self):
        return len(self.data)
    __len__ = _get_length

    def _get_data(self):
        return self._data

    def _set_data(self, data):
        self._data = data
    data = nodeValue = property(_get_data, _set_data)

    def __repr__(self):
        data = self.data
        if len(data) > 10:
            dotdotdot = '...'
        else:
            dotdotdot = ''
        return '<DOM %s node "%r%s">' % (self.__class__.__name__, data[0:10], dotdotdot)

    def substringData(self, offset, count):
        if offset < 0:
            raise xml.dom.IndexSizeErr('offset cannot be negative')
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr('offset cannot be beyond end of data')
        if count < 0:
            raise xml.dom.IndexSizeErr('count cannot be negative')
        return self.data[offset:offset + count]

    def appendData(self, arg):
        self.data = self.data + arg

    def insertData(self, offset, arg):
        if offset < 0:
            raise xml.dom.IndexSizeErr('offset cannot be negative')
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr('offset cannot be beyond end of data')
        if arg:
            self.data = '%s%s%s' % (self.data[:offset], arg, self.data[offset:])

    def deleteData(self, offset, count):
        if offset < 0:
            raise xml.dom.IndexSizeErr('offset cannot be negative')
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr('offset cannot be beyond end of data')
        if count < 0:
            raise xml.dom.IndexSizeErr('count cannot be negative')
        if count:
            self.data = self.data[:offset] + self.data[offset + count:]

    def replaceData(self, offset, count, arg):
        if offset < 0:
            raise xml.dom.IndexSizeErr('offset cannot be negative')
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr('offset cannot be beyond end of data')
        if count < 0:
            raise xml.dom.IndexSizeErr('count cannot be negative')
        if count:
            self.data = '%s%s%s' % (self.data[:offset], arg, self.data[offset + count:])