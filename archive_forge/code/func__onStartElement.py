from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
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