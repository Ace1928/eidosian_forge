from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc5480
from pyasn1_modules import rfc4055
from pyasn1_modules import rfc3279
class RsaSsa_Pss_sig_caps(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('hashAlg', AlgorithmIdentifier()), namedtype.OptionalNamedType('maskAlg', AlgorithmIdentifier()), namedtype.DefaultedNamedType('trailerField', univ.Integer().subtype(value=1)))