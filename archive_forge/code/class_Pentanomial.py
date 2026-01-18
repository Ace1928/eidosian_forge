from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class Pentanomial(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('k1', univ.Integer()), namedtype.NamedType('k2', univ.Integer()), namedtype.NamedType('k3', univ.Integer()))