import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class GraphicString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 25))
    encoding = 'iso-8859-1'
    typeId = AbstractCharacterString.getTypeId()