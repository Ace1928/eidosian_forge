from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class CertID(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('hashAlgorithm', rfc2459.AlgorithmIdentifier()), namedtype.NamedType('issuerNameHash', univ.OctetString()), namedtype.NamedType('issuerKeyHash', univ.OctetString()), namedtype.NamedType('serialNumber', rfc2459.CertificateSerialNumber()))