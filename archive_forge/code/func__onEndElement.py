from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def _onEndElement(self, _):
    if self.currElem is None:
        self.DocumentEndEvent()
    elif self.currElem.parent is None:
        self.ElementEvent(self.currElem)
        self.currElem = None
    else:
        self.currElem = self.currElem.parent