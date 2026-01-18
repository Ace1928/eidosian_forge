from pyasn1.type import char
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class EncryptedPEPSI(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('identifierType', univ.ObjectIdentifier()), namedtype.NamedType('identifier', char.UTF8String()), namedtype.NamedType('sIM', SIM()))