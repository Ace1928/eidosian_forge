import datetime
from pyasn1 import error
from pyasn1.compat import dateandtime
from pyasn1.compat import string
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
class ObjectDescriptor(char.GraphicString):
    __doc__ = char.GraphicString.__doc__
    tagSet = char.GraphicString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 7))
    typeId = char.GraphicString.getTypeId()