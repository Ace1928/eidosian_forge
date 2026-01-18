from pyasn1.type import constraint
from pyasn1.type import namedval
from pyasn1_modules.rfc2437 import *
class OtherPrimeInfos(univ.SequenceOf):
    componentType = OtherPrimeInfo()
    subtypeSpec = univ.SequenceOf.subtypeSpec + constraint.ValueSizeConstraint(1, MAX)