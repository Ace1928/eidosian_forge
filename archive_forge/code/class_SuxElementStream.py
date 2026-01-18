from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
class SuxElementStream(sux.XMLParser):

    def __init__(self):
        self.connectionMade()
        self.DocumentStartEvent = None
        self.ElementEvent = None
        self.DocumentEndEvent = None
        self.currElem = None
        self.rootElem = None
        self.documentStarted = False
        self.defaultNsStack = []
        self.prefixStack = []

    def parse(self, buffer):
        try:
            self.dataReceived(buffer)
        except sux.ParseError as e:
            raise ParserError(str(e))

    def findUri(self, prefix):
        stack = self.prefixStack
        for i in range(-1, (len(self.prefixStack) + 1) * -1, -1):
            if prefix in stack[i]:
                return stack[i][prefix]
        return None

    def gotTagStart(self, name, attributes):
        defaultUri = None
        localPrefixes = {}
        attribs = {}
        uri = None
        for k, v in list(attributes.items()):
            if k.startswith('xmlns'):
                x, p = _splitPrefix(k)
                if x is None:
                    defaultUri = v
                else:
                    localPrefixes[p] = v
                del attributes[k]
        self.prefixStack.append(localPrefixes)
        if defaultUri is None:
            if len(self.defaultNsStack) > 0:
                defaultUri = self.defaultNsStack[-1]
            else:
                defaultUri = ''
        prefix, name = _splitPrefix(name)
        if prefix is None:
            uri = defaultUri
        else:
            uri = self.findUri(prefix)
        for k, v in attributes.items():
            p, n = _splitPrefix(k)
            if p is None:
                attribs[n] = v
            else:
                attribs[self.findUri(p), n] = unescapeFromXml(v)
        e = Element((uri, name), defaultUri, attribs, localPrefixes)
        self.defaultNsStack.append(defaultUri)
        if self.documentStarted:
            if self.currElem is None:
                self.currElem = e
            else:
                self.currElem = self.currElem.addChild(e)
        else:
            self.rootElem = e
            self.documentStarted = True
            self.DocumentStartEvent(e)

    def gotText(self, data):
        if self.currElem is not None:
            if isinstance(data, bytes):
                data = data.decode('ascii')
            self.currElem.addContent(data)

    def gotCData(self, data):
        if self.currElem is not None:
            if isinstance(data, bytes):
                data = data.decode('ascii')
            self.currElem.addContent(data)

    def gotComment(self, data):
        pass
    entities = {'amp': '&', 'lt': '<', 'gt': '>', 'apos': "'", 'quot': '"'}

    def gotEntityReference(self, entityRef):
        if entityRef in SuxElementStream.entities:
            data = SuxElementStream.entities[entityRef]
            if isinstance(data, bytes):
                data = data.decode('ascii')
            self.currElem.addContent(data)

    def gotTagEnd(self, name):
        if self.rootElem is None:
            raise ParserError('Element closed after end of document.')
        prefix, name = _splitPrefix(name)
        if prefix is None:
            uri = self.defaultNsStack[-1]
        else:
            uri = self.findUri(prefix)
        if self.currElem is None:
            if self.rootElem.name != name or self.rootElem.uri != uri:
                raise ParserError('Mismatched root elements')
            self.DocumentEndEvent()
            self.rootElem = None
        else:
            if self.currElem.name != name or self.currElem.uri != uri:
                raise ParserError('Malformed element close')
            self.prefixStack.pop()
            self.defaultNsStack.pop()
            if self.currElem.parent is None:
                self.currElem.parent = self.rootElem
                self.ElementEvent(self.currElem)
                self.currElem = None
            else:
                self.currElem = self.currElem.parent