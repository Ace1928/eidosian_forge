import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ
class UTF8String(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12))
    encoding = 'utf-8'
    typeId = AbstractCharacterString.getTypeId()