from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def _onStartNamespace(self, prefix, uri):
    if prefix is None:
        self.defaultNsStack.append(uri)
    else:
        self.localPrefixes[prefix] = uri