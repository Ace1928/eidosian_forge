from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class _OtherAttr(_Attribute):
    """
    A text attribute for text with formatting attributes.

    The unary minus operator returns the inverse of this attribute, where that
    makes sense.

    @type attrname: C{str}
    @ivar attrname: Text attribute name.

    @ivar attrvalue: Text attribute value.
    """
    compareAttributes = ('attrname', 'attrvalue', 'children')

    def __init__(self, attrname, attrvalue):
        _Attribute.__init__(self)
        self.attrname = attrname
        self.attrvalue = attrvalue

    def __neg__(self):
        result = _OtherAttr(self.attrname, not self.attrvalue)
        result.children.extend(self.children)
        return result

    def serialize(self, write, attrs, attributeRenderer):
        attrs = attrs._withAttribute(self.attrname, self.attrvalue)
        _Attribute.serialize(self, write, attrs, attributeRenderer)