from pyasn1.type import univ, namedtype, tag
class AsnPubKey(univ.Sequence):
    """ASN.1 contents of DER encoded public key:

    RSAPublicKey ::= SEQUENCE {
         modulus           INTEGER,  -- n
         publicExponent    INTEGER,  -- e
    """
    componentType = namedtype.NamedTypes(namedtype.NamedType('modulus', univ.Integer()), namedtype.NamedType('publicExponent', univ.Integer()))