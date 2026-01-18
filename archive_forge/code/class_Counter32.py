from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class Counter32(univ.Integer):
    tagSet = univ.Integer.tagSet.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 1))
    subtypeSpec = univ.Integer.subtypeSpec + constraint.ValueRangeConstraint(0, 4294967295)