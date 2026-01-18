from pyasn1.type import char
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5755
class Caterpillar_SecurityClassification(univ.Integer):
    namedValues = namedval.NamedValues(('caterpillar-public', 6), ('caterpillar-green', 7), ('caterpillar-yellow', 8), ('caterpillar-red', 9))