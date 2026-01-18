from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class SimpleSyntax(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('number', univ.Integer()), namedtype.NamedType('string', univ.OctetString()), namedtype.NamedType('object', univ.ObjectIdentifier()), namedtype.NamedType('empty', univ.Null()))