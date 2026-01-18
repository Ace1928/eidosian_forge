import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def _save_pkcs1_der(self):
    """Saves the private key in PKCS#1 DER format.

        :returns: the DER-encoded private key.
        :rtype: bytes
        """
    from pyasn1.type import univ, namedtype
    from pyasn1.codec.der import encoder

    class AsnPrivKey(univ.Sequence):
        componentType = namedtype.NamedTypes(namedtype.NamedType('version', univ.Integer()), namedtype.NamedType('modulus', univ.Integer()), namedtype.NamedType('publicExponent', univ.Integer()), namedtype.NamedType('privateExponent', univ.Integer()), namedtype.NamedType('prime1', univ.Integer()), namedtype.NamedType('prime2', univ.Integer()), namedtype.NamedType('exponent1', univ.Integer()), namedtype.NamedType('exponent2', univ.Integer()), namedtype.NamedType('coefficient', univ.Integer()))
    asn_key = AsnPrivKey()
    asn_key.setComponentByName('version', 0)
    asn_key.setComponentByName('modulus', self.n)
    asn_key.setComponentByName('publicExponent', self.e)
    asn_key.setComponentByName('privateExponent', self.d)
    asn_key.setComponentByName('prime1', self.p)
    asn_key.setComponentByName('prime2', self.q)
    asn_key.setComponentByName('exponent1', self.exp1)
    asn_key.setComponentByName('exponent2', self.exp2)
    asn_key.setComponentByName('coefficient', self.coef)
    return encoder.encode(asn_key)