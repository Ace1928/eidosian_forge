from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class SearchRequest(univ.Sequence):
    tagSet = univ.Sequence.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 3))
    componentType = namedtype.NamedTypes(namedtype.NamedType('baseObject', LDAPDN()), namedtype.NamedType('scope', univ.Enumerated(namedValues=namedval.NamedValues(('baseObject', 0), ('singleLevel', 1), ('wholeSubtree', 2)))), namedtype.NamedType('derefAliases', univ.Enumerated(namedValues=namedval.NamedValues(('neverDerefAliases', 0), ('derefInSearching', 1), ('derefFindingBaseObj', 2), ('derefAlways', 3)))), namedtype.NamedType('sizeLimit', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, maxInt))), namedtype.NamedType('timeLimit', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, maxInt))), namedtype.NamedType('typesOnly', univ.Boolean()), namedtype.NamedType('filter', Filter()), namedtype.NamedType('attributes', AttributeDescriptionList()))