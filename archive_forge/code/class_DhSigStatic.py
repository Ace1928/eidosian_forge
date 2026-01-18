from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc3279
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
class DhSigStatic(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('issuerAndSerial', IssuerAndSerialNumber()), namedtype.NamedType('hashValue', MessageDigest()))