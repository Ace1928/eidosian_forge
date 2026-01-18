from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class ModifyRequest(univ.Sequence):
    tagSet = univ.Sequence.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatConstructed, 6))
    componentType = namedtype.NamedTypes(namedtype.NamedType('object', LDAPDN()), namedtype.NamedType('modification', univ.SequenceOf(componentType=univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('operation', univ.Enumerated(namedValues=namedval.NamedValues(('add', 0), ('delete', 1), ('replace', 2)))), namedtype.NamedType('modification', AttributeTypeAndValues()))))))