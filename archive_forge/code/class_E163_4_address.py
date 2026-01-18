from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class E163_4_address(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('number', char.NumericString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_e163_4_number_length), explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('sub-address', char.NumericString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, ub_e163_4_sub_address_length), explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))))