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
class KeyRecRepContent(univ.Sequence):
    """
    KeyRecRepContent ::= SEQUENCE {
         status                  PKIStatusInfo,
         newSigCert          [0] CMPCertificate OPTIONAL,
         caCerts             [1] SEQUENCE SIZE (1..MAX) OF
                                             CMPCertificate OPTIONAL,
         keyPairHist         [2] SEQUENCE SIZE (1..MAX) OF
                                             CertifiedKeyPair OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('status', PKIStatusInfo()), namedtype.OptionalNamedType('newSigCert', CMPCertificate().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))), namedtype.OptionalNamedType('caCerts', univ.SequenceOf(componentType=CMPCertificate()).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), subtypeSpec=constraint.ValueSizeConstraint(1, MAX))), namedtype.OptionalNamedType('keyPairHist', univ.SequenceOf(componentType=CertifiedKeyPair()).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2), subtypeSpec=constraint.ValueSizeConstraint(1, MAX))))