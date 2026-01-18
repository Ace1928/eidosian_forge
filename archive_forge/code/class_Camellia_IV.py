from pyasn1.type import constraint
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5751
class Camellia_IV(univ.OctetString):
    subtypeSpec = constraint.ValueSizeConstraint(16, 16)