from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc4357
from pyasn1_modules import rfc5280
class GostR3410_2001_Signature(univ.OctetString):
    subtypeSpec = constraint.ValueSizeConstraint(64, 64)