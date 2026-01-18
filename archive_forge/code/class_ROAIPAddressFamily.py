from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5652
class ROAIPAddressFamily(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('addressFamily', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(2, 3))), namedtype.NamedType('addresses', univ.SequenceOf(componentType=ROAIPAddress()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))))