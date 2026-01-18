from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class Characteristic_two(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('m', univ.Integer()), namedtype.NamedType('basis', univ.ObjectIdentifier()), namedtype.NamedType('parameters', univ.Any()))