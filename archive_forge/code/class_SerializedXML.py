from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
class SerializedXML(str):
    """Marker class for pre-serialized XML in the DOM."""
    pass