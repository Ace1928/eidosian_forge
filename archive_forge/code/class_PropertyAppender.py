from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class PropertyAppender(Appender):
    """
    A L{Property} appender.
    """

    def append(self, parent, content):
        p = content.value
        child = self.node(content)
        child.setText(p.get())
        parent.append(child)
        for item in list(p.items()):
            cont = Content(tag=item[0], value=item[1])
            Appender.append(self, child, cont)