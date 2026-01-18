from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc1905
class ScopedPDU(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('contextEngineId', univ.OctetString()), namedtype.NamedType('contextName', univ.OctetString()), namedtype.NamedType('data', rfc1905.PDUs()))