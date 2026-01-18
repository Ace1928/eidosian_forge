from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def addRawXml(self, rawxmlstring):
    """Add a pre-serialized chunk o' XML as a child of this Element."""
    self.children.append(SerializedXML(rawxmlstring))