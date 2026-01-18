from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class ResponderID(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('byName', rfc2459.Name().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('byKey', KeyHash().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))