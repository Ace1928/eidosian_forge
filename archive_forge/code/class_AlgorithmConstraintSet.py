from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class AlgorithmConstraintSet(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('signerAlgorithmConstraints', AlgorithmConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('eeCertAlgorithmConstraints', AlgorithmConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.OptionalNamedType('caCertAlgorithmConstraints', AlgorithmConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))), namedtype.OptionalNamedType('aaCertAlgorithmConstraints', AlgorithmConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))), namedtype.OptionalNamedType('tsaCertAlgorithmConstraints', AlgorithmConstraints().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))))