from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc1905
class HeaderData(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('msgID', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, 2147483647))), namedtype.NamedType('msgMaxSize', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(484, 2147483647))), namedtype.NamedType('msgFlags', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 1))), namedtype.NamedType('msgSecurityModel', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, 2147483647))))