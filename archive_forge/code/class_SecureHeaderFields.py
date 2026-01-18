from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5652
import string
class SecureHeaderFields(univ.Set):
    componentType = namedtype.NamedTypes(namedtype.NamedType('canonAlgorithm', Algorithm()), namedtype.NamedType('secHeaderFields', HeaderFields()))