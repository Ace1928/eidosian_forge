from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class Pkcs9email(char.IA5String):
    subtypeSpec = char.IA5String.subtypeSpec + constraint.ValueSizeConstraint(1, ub_emailaddress_length)