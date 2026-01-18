from pyasn1.type import univ, constraint, namedtype, namedval, tag
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
class OneAsymmetricKey(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', Version()), namedtype.NamedType('privateKeyAlgorithm', PrivateKeyAlgorithmIdentifier()), namedtype.NamedType('privateKey', PrivateKey()), namedtype.OptionalNamedType('attributes', Attributes().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('publicKey', PublicKey().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))))