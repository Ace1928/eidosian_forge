from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class EcpkParameters(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('ecParameters', ECParameters()), namedtype.NamedType('namedCurve', univ.ObjectIdentifier()), namedtype.NamedType('implicitlyCA', univ.Null()))