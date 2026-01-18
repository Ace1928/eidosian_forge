from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful
from pyasn1_modules import rfc2560
from pyasn1_modules import rfc5280
class PreferredSignatureAlgorithms(univ.SequenceOf):
    componentType = PreferredSignatureAlgorithm()