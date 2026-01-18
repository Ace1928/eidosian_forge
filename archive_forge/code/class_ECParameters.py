from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class ECParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', ECPVer()), namedtype.NamedType('fieldID', FieldID()), namedtype.NamedType('curve', Curve()), namedtype.NamedType('base', ECPoint()), namedtype.NamedType('order', univ.Integer()), namedtype.OptionalNamedType('cofactor', univ.Integer()))