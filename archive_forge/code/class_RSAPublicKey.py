from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class RSAPublicKey(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('modulus', univ.Integer()), namedtype.NamedType('publicExponent', univ.Integer()))