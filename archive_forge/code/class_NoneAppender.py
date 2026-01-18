from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class NoneAppender(Appender):
    """
    An appender for I{None} values.
    """

    def append(self, parent, content):
        child = self.node(content)
        default = self.setdefault(child, content)
        if default is None:
            self.setnil(child, content)
        parent.append(child)