from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class POPOPrivKey(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('thisMessage', univ.BitString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('subsequentMessage', SubsequentMessage().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('dhMAC', univ.BitString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))