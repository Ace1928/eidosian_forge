from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def addUniqueId(self):
    """Add a unique (across a given Python session) id attribute to this
        Element.
        """
    self.attributes['id'] = 'H_%d' % Element._idCounter
    Element._idCounter = Element._idCounter + 1