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
class DHBMParameter(univ.Sequence):
    """
    DHBMParameter ::= SEQUENCE {
         owf                 AlgorithmIdentifier,
         -- AlgId for a One-Way Function (SHA-1 recommended)
         mac                 AlgorithmIdentifier
         -- the MAC AlgId (e.g., DES-MAC, Triple-DES-MAC [PKCS11],
     }   -- or HMAC [RFC2104, RFC2202])
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('owf', rfc2459.AlgorithmIdentifier()), namedtype.NamedType('mac', rfc2459.AlgorithmIdentifier()))