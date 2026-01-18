from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5751
class Skipjack_Parm(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('initialization-vector', univ.OctetString()))