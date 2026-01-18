from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class TextAppender(Appender):
    """
    An appender for I{Text} values.
    """

    def append(self, parent, content):
        if content.tag.startswith('_') and isinstance(content.type, Attribute):
            attr = content.tag[1:]
            value = tostr(content.value)
            if value:
                parent.set(attr, value)
        else:
            child = self.node(content)
            child.setText(content.value)
            parent.append(child)