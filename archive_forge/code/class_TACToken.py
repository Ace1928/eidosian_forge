from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc5652
class TACToken(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('userKey', UserKey()), namedtype.NamedType('timeout', Timeout()))