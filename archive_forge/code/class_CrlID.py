from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc2560
from pyasn1_modules import rfc5280
class CrlID(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('crlUrl', char.IA5String().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('crlNum', univ.Integer().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.OptionalNamedType('crlTime', useful.GeneralizedTime().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))