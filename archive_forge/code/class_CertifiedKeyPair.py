from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2314
from pyasn1_modules import rfc2459
from pyasn1_modules import rfc2511
class CertifiedKeyPair(univ.Sequence):
    """
    CertifiedKeyPair ::= SEQUENCE {
         certOrEncCert       CertOrEncCert,
         privateKey      [0] EncryptedValue      OPTIONAL,
         publicationInfo [1] PKIPublicationInfo  OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('certOrEncCert', CertOrEncCert()), namedtype.OptionalNamedType('privateKey', rfc2511.EncryptedValue().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('publicationInfo', rfc2511.PKIPublicationInfo().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))))