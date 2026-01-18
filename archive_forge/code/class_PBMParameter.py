from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class PBMParameter(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('salt', univ.OctetString()), namedtype.NamedType('owf', AlgorithmIdentifier()), namedtype.NamedType('iterationCount', univ.Integer()), namedtype.NamedType('mac', AlgorithmIdentifier()))