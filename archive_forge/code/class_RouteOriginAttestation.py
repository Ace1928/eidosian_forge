from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5652
class RouteOriginAttestation(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.DefaultedNamedType('version', univ.Integer().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)).subtype(value=0)), namedtype.NamedType('asID', ASID()), namedtype.NamedType('ipAddrBlocks', univ.SequenceOf(componentType=ROAIPAddressFamily()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))))