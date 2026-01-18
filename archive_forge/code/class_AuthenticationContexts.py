from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class AuthenticationContexts(univ.SequenceOf):
    componentType = AuthenticationContext()
    subtypeSpec = constraint.ValueSizeConstraint(1, MAX)