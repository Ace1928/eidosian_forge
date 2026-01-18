from Cryptodome.Util.asn1 import (DerSequence, DerInteger, DerBitString,
def _create_subject_public_key_info(algo_oid, public_key, params):
    if params is None:
        algorithm = DerSequence([DerObjectId(algo_oid)])
    else:
        algorithm = DerSequence([DerObjectId(algo_oid), params])
    spki = DerSequence([algorithm, DerBitString(public_key)])
    return spki.encode()