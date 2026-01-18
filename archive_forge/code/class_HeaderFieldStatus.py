from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5652
import string
class HeaderFieldStatus(univ.Integer):
    namedValues = namedval.NamedValues(('duplicated', 0), ('deleted', 1), ('modified', 2))