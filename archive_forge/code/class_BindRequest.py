from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class BindRequest(univ.Sequence):
    tagSet = univ.Sequence.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 0))
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, 127))), namedtype.NamedType('name', LDAPDN()), namedtype.NamedType('authentication', AuthenticationChoice()))