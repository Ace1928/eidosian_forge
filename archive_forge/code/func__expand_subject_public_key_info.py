from Cryptodome.Util.asn1 import (DerSequence, DerInteger, DerBitString,
def _expand_subject_public_key_info(encoded):
    """Parse a SubjectPublicKeyInfo structure.

    It returns a triple with:
        * OID (string)
        * encoded public key (bytes)
        * Algorithm parameters (bytes or None)
    """
    spki = DerSequence().decode(encoded, nr_elements=2)
    algo = DerSequence().decode(spki[0], nr_elements=(1, 2))
    algo_oid = DerObjectId().decode(algo[0])
    spk = DerBitString().decode(spki[1]).value
    if len(algo) == 1:
        algo_params = None
    else:
        try:
            DerNull().decode(algo[1])
            algo_params = None
        except:
            algo_params = algo[1]
    return (algo_oid.value, spk, algo_params)