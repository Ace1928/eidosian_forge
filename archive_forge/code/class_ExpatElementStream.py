from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
class ExpatElementStream:

    def __init__(self):
        import pyexpat
        self.DocumentStartEvent = None
        self.ElementEvent = None
        self.DocumentEndEvent = None
        self.error = pyexpat.error
        self.parser = pyexpat.ParserCreate('UTF-8', ' ')
        self.parser.StartElementHandler = self._onStartElement
        self.parser.EndElementHandler = self._onEndElement
        self.parser.CharacterDataHandler = self._onCdata
        self.parser.StartNamespaceDeclHandler = self._onStartNamespace
        self.parser.EndNamespaceDeclHandler = self._onEndNamespace
        self.currElem = None
        self.defaultNsStack = ['']
        self.documentStarted = 0
        self.localPrefixes = {}

    def parse(self, buffer):
        try:
            self.parser.Parse(buffer)
        except self.error as e:
            raise ParserError(str(e))

    def _onStartElement(self, name, attrs):
        qname = name.rsplit(' ', 1)
        if len(qname) == 1:
            qname = ('', name)
        newAttrs = {}
        toDelete = []
        for k, v in attrs.items():
            if ' ' in k:
                aqname = k.rsplit(' ', 1)
                newAttrs[aqname[0], aqname[1]] = v
                toDelete.append(k)
        attrs.update(newAttrs)
        for k in toDelete:
            del attrs[k]
        e = Element(qname, self.defaultNsStack[-1], attrs, self.localPrefixes)
        self.localPrefixes = {}
        if self.documentStarted == 1:
            if self.currElem != None:
                self.currElem.children.append(e)
                e.parent = self.currElem
            self.currElem = e
        else:
            self.documentStarted = 1
            self.DocumentStartEvent(e)

    def _onEndElement(self, _):
        if self.currElem is None:
            self.DocumentEndEvent()
        elif self.currElem.parent is None:
            self.ElementEvent(self.currElem)
            self.currElem = None
        else:
            self.currElem = self.currElem.parent

    def _onCdata(self, data):
        if self.currElem != None:
            self.currElem.addContent(data)

    def _onStartNamespace(self, prefix, uri):
        if prefix is None:
            self.defaultNsStack.append(uri)
        else:
            self.localPrefixes[prefix] = uri

    def _onEndNamespace(self, prefix):
        if prefix is None:
            self.defaultNsStack.pop()