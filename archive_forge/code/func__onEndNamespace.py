from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def _onEndNamespace(self, prefix):
    if prefix is None:
        self.defaultNsStack.pop()