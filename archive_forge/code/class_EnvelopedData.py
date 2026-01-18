from pyasn1_modules.rfc2459 import *
class EnvelopedData(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', Version()), namedtype.NamedType('recipientInfos', RecipientInfos()), namedtype.NamedType('encryptedContentInfo', EncryptedContentInfo()))