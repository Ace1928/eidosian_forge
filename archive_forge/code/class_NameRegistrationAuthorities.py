from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc5280
class NameRegistrationAuthorities(univ.SequenceOf):
    componentType = GeneralName()
    subtypeSpec = constraint.ValueSizeConstraint(1, MAX)