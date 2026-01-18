from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
class _ListSerializer:
    """Internal class which serializes an Element tree into a buffer"""

    def __init__(self, prefixes=None, prefixesInScope=None):
        self.writelist = []
        self.prefixes = {}
        if prefixes:
            self.prefixes.update(prefixes)
        self.prefixes.update(G_PREFIXES)
        self.prefixStack = [G_PREFIXES.values()] + (prefixesInScope or [])
        self.prefixCounter = 0

    def getValue(self):
        return ''.join(self.writelist)

    def getPrefix(self, uri):
        if uri not in self.prefixes:
            self.prefixes[uri] = 'xn%d' % self.prefixCounter
            self.prefixCounter = self.prefixCounter + 1
        return self.prefixes[uri]

    def prefixInScope(self, prefix):
        stack = self.prefixStack
        for i in range(-1, (len(self.prefixStack) + 1) * -1, -1):
            if prefix in stack[i]:
                return True
        return False

    def serialize(self, elem, closeElement=1, defaultUri=''):
        write = self.writelist.append
        if isinstance(elem, SerializedXML):
            write(elem)
            return
        if isinstance(elem, str):
            write(escapeToXml(elem))
            return
        name = elem.name
        uri = elem.uri
        defaultUri, currentDefaultUri = (elem.defaultUri, defaultUri)
        for p, u in elem.localPrefixes.items():
            self.prefixes[u] = p
        self.prefixStack.append(list(elem.localPrefixes.keys()))
        if defaultUri is None:
            defaultUri = currentDefaultUri
        if uri is None:
            uri = defaultUri
        prefix = None
        if uri != defaultUri or uri in self.prefixes:
            prefix = self.getPrefix(uri)
            inScope = self.prefixInScope(prefix)
        if not prefix:
            write('<%s' % name)
        else:
            write(f'<{prefix}:{name}')
            if not inScope:
                write(f" xmlns:{prefix}='{uri}'")
                self.prefixStack[-1].append(prefix)
                inScope = True
        if defaultUri != currentDefaultUri and (uri != defaultUri or not prefix or (not inScope)):
            write(" xmlns='%s'" % defaultUri)
        for p, u in elem.localPrefixes.items():
            write(f" xmlns:{p}='{u}'")
        for k, v in elem.attributes.items():
            if isinstance(k, tuple):
                attr_uri, attr_name = k
                attr_prefix = self.getPrefix(attr_uri)
                if not self.prefixInScope(attr_prefix):
                    write(f" xmlns:{attr_prefix}='{attr_uri}'")
                    self.prefixStack[-1].append(attr_prefix)
                write(f" {attr_prefix}:{attr_name}='{escapeToXml(v, 1)}'")
            else:
                write(f" {k}='{escapeToXml(v, 1)}'")
        if closeElement == 0:
            write('>')
            return
        if len(elem.children) > 0:
            write('>')
            for c in elem.children:
                self.serialize(c, defaultUri=defaultUri)
            if not prefix:
                write('</%s>' % name)
            else:
                write(f'</{prefix}:{name}>')
        else:
            write('/>')
        self.prefixStack.pop()