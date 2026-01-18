from pyasn1_modules import rfc2251
from pyasn1_modules.rfc2459 import *
class EncryptedPrivateKeyInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('encryptionAlgorithm', AlgorithmIdentifier()), namedtype.NamedType('encryptedData', EncryptedData()))