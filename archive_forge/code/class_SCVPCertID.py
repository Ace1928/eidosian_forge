from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc4055
class SCVPCertID(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('certHash', univ.OctetString()), namedtype.NamedType('issuerSerial', SCVPIssuerSerial()), namedtype.DefaultedNamedType('hashAlgorithm', sha1_alg_id))