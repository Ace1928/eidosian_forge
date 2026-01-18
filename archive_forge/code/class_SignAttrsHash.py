from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5035
from pyasn1_modules import rfc5652
class SignAttrsHash(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('algID', rfc5652.DigestAlgorithmIdentifier()), namedtype.NamedType('hash', univ.OctetString()))