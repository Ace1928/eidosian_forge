from pyasn1.type import constraint
from pyasn1.type import namedval
from pyasn1_modules.rfc2437 import *
class RSAPrivateKey(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('version', univ.Integer(namedValues=namedval.NamedValues(('two-prime', 0), ('multi', 1)))), namedtype.NamedType('modulus', univ.Integer()), namedtype.NamedType('publicExponent', univ.Integer()), namedtype.NamedType('privateExponent', univ.Integer()), namedtype.NamedType('prime1', univ.Integer()), namedtype.NamedType('prime2', univ.Integer()), namedtype.NamedType('exponent1', univ.Integer()), namedtype.NamedType('exponent2', univ.Integer()), namedtype.NamedType('coefficient', univ.Integer()), namedtype.OptionalNamedType('otherPrimeInfos', OtherPrimeInfos()))