import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
class AsnPrivKey(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', univ.Integer()), namedtype.NamedType('modulus', univ.Integer()), namedtype.NamedType('publicExponent', univ.Integer()), namedtype.NamedType('privateExponent', univ.Integer()), namedtype.NamedType('prime1', univ.Integer()), namedtype.NamedType('prime2', univ.Integer()), namedtype.NamedType('exponent1', univ.Integer()), namedtype.NamedType('exponent2', univ.Integer()), namedtype.NamedType('coefficient', univ.Integer()))