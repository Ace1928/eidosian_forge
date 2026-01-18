from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class NAIRealm(char.UTF8String):
    subtypeSpec = constraint.ValueSizeConstraint(1, ub_naiRealm_length)