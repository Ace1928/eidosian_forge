from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class ElementAppender(Appender):
    """
    An appender for I{Element} types.
    """

    def append(self, parent, content):
        if content.tag.startswith('_') and isinstance(content.type, Attribute):
            raise Exception('raw XML not valid as attribute value')
        child = ElementWrapper(content.value)
        parent.append(child)