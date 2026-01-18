from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class POPOSigningKey(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('poposkInput', POPOSigningKeyInput().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.NamedType('algorithmIdentifier', AlgorithmIdentifier()), namedtype.NamedType('signature', univ.BitString()))