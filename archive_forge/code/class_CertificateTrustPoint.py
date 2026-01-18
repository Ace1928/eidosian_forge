from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class CertificateTrustPoint(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('trustpoint', Certificate()), namedtype.OptionalNamedType('pathLenConstraint', PathLenConstraint().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('acceptablePolicySet', AcceptablePolicySet().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.OptionalNamedType('nameConstraints', NameConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))), namedtype.OptionalNamedType('policyConstraints', PolicyConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))))