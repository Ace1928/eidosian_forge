from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class CompareRequest(univ.Sequence):
    tagSet = univ.Sequence.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 14))
    componentType = namedtype.NamedTypes(namedtype.NamedType('entry', LDAPDN()), namedtype.NamedType('ava', AttributeValueAssertion()))