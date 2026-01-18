from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class PKIArchiveOptions(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('encryptedPrivKey', EncryptedKey().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.NamedType('keyGenParameters', KeyGenParameters().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('archiveRemGenPrivKey', univ.Boolean().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))