from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class FieldID(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('fieldType', univ.ObjectIdentifier()), namedtype.NamedType('parameters', univ.Any()))