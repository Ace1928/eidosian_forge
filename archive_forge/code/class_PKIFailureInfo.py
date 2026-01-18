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
class PKIFailureInfo(univ.BitString):
    """
    PKIFailureInfo ::= BIT STRING {
         badAlg              (0),
         badMessageCheck     (1),
         badRequest          (2),
         badTime             (3),
         badCertId           (4),
         badDataFormat       (5),
         wrongAuthority      (6),
         incorrectData       (7),
         missingTimeStamp    (8),
         badPOP              (9),
         certRevoked         (10),
         certConfirmed       (11),
         wrongIntegrity      (12),
         badRecipientNonce   (13),
         timeNotAvailable    (14),
         unacceptedPolicy    (15),
         unacceptedExtension (16),
         addInfoNotAvailable (17),
         badSenderNonce      (18),
         badCertTemplate     (19),
         signerNotTrusted    (20),
         transactionIdInUse  (21),
         unsupportedVersion  (22),
         notAuthorized       (23),
         systemUnavail       (24),
         systemFailure       (25),
         duplicateCertReq    (26)
    """
    namedValues = namedval.NamedValues(('badAlg', 0), ('badMessageCheck', 1), ('badRequest', 2), ('badTime', 3), ('badCertId', 4), ('badDataFormat', 5), ('wrongAuthority', 6), ('incorrectData', 7), ('missingTimeStamp', 8), ('badPOP', 9), ('certRevoked', 10), ('certConfirmed', 11), ('wrongIntegrity', 12), ('badRecipientNonce', 13), ('timeNotAvailable', 14), ('unacceptedPolicy', 15), ('unacceptedExtension', 16), ('addInfoNotAvailable', 17), ('badSenderNonce', 18), ('badCertTemplate', 19), ('signerNotTrusted', 20), ('transactionIdInUse', 21), ('unsupportedVersion', 22), ('notAuthorized', 23), ('systemUnavail', 24), ('systemFailure', 25), ('duplicateCertReq', 26))