from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class AuthenticationChoice(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('simple', univ.OctetString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('reserved-1', univ.OctetString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('reserved-2', univ.OctetString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))), namedtype.NamedType('sasl', SaslCredentials().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))))