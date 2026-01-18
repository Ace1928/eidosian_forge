from pyasn1.type import constraint
from pyasn1.type import namedval
from pyasn1_modules.rfc2437 import *
class OtherPrimeInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('prime', univ.Integer()), namedtype.NamedType('exponent', univ.Integer()), namedtype.NamedType('coefficient', univ.Integer()))