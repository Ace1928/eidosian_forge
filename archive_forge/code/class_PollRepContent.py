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
class PollRepContent(univ.SequenceOf):
    """
         PollRepContent ::= SEQUENCE OF SEQUENCE {
         certReqId              INTEGER,
         checkAfter             INTEGER,  -- time in seconds
         reason                 PKIFreeText OPTIONAL
     }
    """

    class CertReq(univ.Sequence):
        componentType = namedtype.NamedTypes(namedtype.NamedType('certReqId', univ.Integer()), namedtype.NamedType('checkAfter', univ.Integer()), namedtype.OptionalNamedType('reason', PKIFreeText()))
    componentType = CertReq()