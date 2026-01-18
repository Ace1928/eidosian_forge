from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
class AttributesNSImpl(AttributesImpl):

    def __init__(self, attrs, qnames):
        """NS-aware implementation.

        attrs should be of the form {(ns_uri, lname): value, ...}.
        qnames of the form {(ns_uri, lname): qname, ...}."""
        self._attrs = attrs
        self._qnames = qnames

    def getValueByQName(self, name):
        for nsname, qname in self._qnames.items():
            if qname == name:
                return self._attrs[nsname]
        raise KeyError(name)

    def getNameByQName(self, name):
        for nsname, qname in self._qnames.items():
            if qname == name:
                return nsname
        raise KeyError(name)

    def getQNameByName(self, name):
        return self._qnames[name]

    def getQNames(self):
        return list(self._qnames.values())

    def copy(self):
        return self.__class__(self._attrs, self._qnames)