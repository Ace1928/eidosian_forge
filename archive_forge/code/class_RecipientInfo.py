from pyasn1_modules.rfc2459 import *
class RecipientInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', Version()), namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()), namedtype.NamedType('keyEncryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()), namedtype.NamedType('encryptedKey', EncryptedKey()))