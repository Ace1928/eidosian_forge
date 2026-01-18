from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class EncryptedKey(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('encryptedValue', EncryptedValue()), namedtype.NamedType('envelopedData', rfc2315.EnvelopedData().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))))