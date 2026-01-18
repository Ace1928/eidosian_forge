from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5652
class KeyAgreePSKRecipientInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', rfc5652.CMSVersion()), namedtype.NamedType('pskid', PreSharedKeyIdentifier()), namedtype.NamedType('originator', rfc5652.OriginatorIdentifierOrKey().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('ukm', rfc5652.UserKeyingMaterial().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('kdfAlgorithm', rfc5652.KeyDerivationAlgorithmIdentifier()), namedtype.NamedType('keyEncryptionAlgorithm', rfc5652.KeyEncryptionAlgorithmIdentifier()), namedtype.NamedType('recipientEncryptedKeys', rfc5652.RecipientEncryptedKeys()))