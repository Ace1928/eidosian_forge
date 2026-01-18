from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class AttributeTrustCondition(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('attributeMandated', univ.Boolean()), namedtype.NamedType('howCertAttribute', HowCertAttribute()), namedtype.OptionalNamedType('attrCertificateTrustTrees', CertificateTrustTrees().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('attrRevReq', CertRevReq().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))), namedtype.OptionalNamedType('attributeConstraints', AttributeConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))))