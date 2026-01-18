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
class PKIStatus(univ.Integer):
    """
    PKIStatus ::= INTEGER {
         accepted                (0),
         grantedWithMods        (1),
         rejection              (2),
         waiting                (3),
         revocationWarning      (4),
         revocationNotification (5),
         keyUpdateWarning       (6)
     }
    """
    namedValues = namedval.NamedValues(('accepted', 0), ('grantedWithMods', 1), ('rejection', 2), ('waiting', 3), ('revocationWarning', 4), ('revocationNotification', 5), ('keyUpdateWarning', 6))