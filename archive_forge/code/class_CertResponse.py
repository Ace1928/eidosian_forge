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
class CertResponse(univ.Sequence):
    """
    CertResponse ::= SEQUENCE {
         certReqId           INTEGER,
         status              PKIStatusInfo,
         certifiedKeyPair    CertifiedKeyPair    OPTIONAL,
         rspInfo             OCTET STRING        OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('certReqId', univ.Integer()), namedtype.NamedType('status', PKIStatusInfo()), namedtype.OptionalNamedType('certifiedKeyPair', CertifiedKeyPair()), namedtype.OptionalNamedType('rspInfo', univ.OctetString()))