from pyasn1.type import char
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5755
class Whirlpool_SecurityClassification(univ.Integer):
    namedValues = namedval.NamedValues(('whirlpool-public', 6), ('whirlpool-internal', 7), ('whirlpool-confidential', 8))