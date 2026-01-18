from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class Scrypt_params(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('salt', univ.OctetString()), namedtype.NamedType('costParameter', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))), namedtype.NamedType('blockSize', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))), namedtype.NamedType('parallelizationParameter', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))), namedtype.OptionalNamedType('keyLength', univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))))