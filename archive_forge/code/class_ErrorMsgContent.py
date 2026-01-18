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
class ErrorMsgContent(univ.Sequence):
    """
    ErrorMsgContent ::= SEQUENCE {
         pKIStatusInfo          PKIStatusInfo,
         errorCode              INTEGER           OPTIONAL,
         -- implementation-specific error codes
         errorDetails           PKIFreeText       OPTIONAL
         -- implementation-specific error details
     }
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('pKIStatusInfo', PKIStatusInfo()), namedtype.OptionalNamedType('errorCode', univ.Integer()), namedtype.OptionalNamedType('errorDetails', PKIFreeText()))