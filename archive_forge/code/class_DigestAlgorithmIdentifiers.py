from pyasn1_modules.rfc2459 import *
class DigestAlgorithmIdentifiers(univ.SetOf):
    componentType = DigestAlgorithmIdentifier()