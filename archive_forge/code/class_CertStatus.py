from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class CertStatus(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('good', univ.Null().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.NamedType('revoked', RevokedInfo().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('unknown', UnknownInfo().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))