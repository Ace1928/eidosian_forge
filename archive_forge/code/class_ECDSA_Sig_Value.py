from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
class ECDSA_Sig_Value(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('r', univ.Integer()), namedtype.NamedType('s', univ.Integer()))