from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc5280
class Gender(char.PrintableString):
    subtypeSpec = constraint.ConstraintsIntersection(constraint.ValueSizeConstraint(1, 1), constraint.SingleValueConstraint('M', 'F', 'm', 'f'))