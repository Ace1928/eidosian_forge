import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class UniversalString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 28))
    encoding = 'utf-32-be'
    typeId = AbstractCharacterString.getTypeId()