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
class RevAnnContent(univ.Sequence):
    """
    RevAnnContent ::= SEQUENCE {
         status              PKIStatus,
         certId              CertId,
         willBeRevokedAt     GeneralizedTime,
         badSinceDate        GeneralizedTime,
         crlDetails          Extensions  OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('status', PKIStatus()), namedtype.NamedType('certId', rfc2511.CertId()), namedtype.NamedType('willBeRevokedAt', useful.GeneralizedTime()), namedtype.NamedType('badSinceDate', useful.GeneralizedTime()), namedtype.OptionalNamedType('crlDetails', rfc2459.Extensions()))